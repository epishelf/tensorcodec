//! FFmpeg handles are owned here and never exposed to Python. Each decoder is
//! exclusively borrowed by PyO3 during a native operation; no global mutable state.
use ffmpeg_sys_next as av;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
use std::ffi::{c_void, CStr, CString};
use std::ptr;

// One reference holds either the selected frame or its lookahead; released on errors too.
struct TimestampCursor {
    frame: *mut av::AVFrame,
    initialized: bool,
    eof: bool,
}
impl TimestampCursor {
    fn new() -> Result<Self> {
        let frame = unsafe { av::av_frame_alloc() };
        if frame.is_null() {
            return Err(failure("cannot allocate lookahead frame"));
        }
        Ok(Self {
            frame,
            initialized: false,
            eof: false,
        })
    }
}
impl Drop for TimestampCursor {
    fn drop(&mut self) {
        unsafe {
            av::av_frame_free(&mut self.frame);
        }
    }
}

pub enum VideoRequest {
    Frames {
        targets: Vec<(i64, i64)>,
        exact: bool,
    },
    Timestamps(Vec<f64>),
}

/// A geometric step; a rotation (counterclockwise quarter turns) comes first, then crops and resizes in the
/// rotated frame's coordinates. `Pipeline` decides which run on the decoded planes and which on RGB.
#[derive(Clone, Copy, PartialEq)]
pub enum Op {
    Rotate {
        turns: i32,
    },
    Crop {
        top: i32,
        left: i32,
        height: i32,
        width: i32,
    },
    Resize {
        height: i32,
        width: i32,
    },
}

/// One resize step: a bilinear swscale pass from a source region straight to RGB at the output size. swscale
/// writes planar RGB with SIMD (its packed full-chroma writer is C only), interleaved into `frame` afterwards.
struct Resizer {
    scale: *mut av::SwsContext,
    config: [i32; 8],
    planar: *mut av::AVFrame,
    frame: *mut av::AVFrame,
}
impl Resizer {
    fn new() -> Result<Self> {
        let (planar, frame) = unsafe { (av::av_frame_alloc(), av::av_frame_alloc()) };
        // Owned before the check, so Drop frees whichever allocation succeeded.
        let resizer = Self {
            scale: ptr::null_mut(),
            config: [0; 8],
            planar,
            frame,
        };
        if planar.is_null() || frame.is_null() {
            return Err(failure("cannot allocate resize frames"));
        }
        Ok(resizer)
    }
}
impl Drop for Resizer {
    fn drop(&mut self) {
        unsafe {
            av::sws_freeContext(self.scale);
            av::av_frame_free(&mut self.planar);
            av::av_frame_free(&mut self.frame);
        }
    }
}

/// Planes of a region to resize: decoded YUV/RGB planes or an RGB intermediate. `chroma_position` is swscale's
/// `src_h_chr_pos`/`src_v_chr_pos` (1/256 luma pixel; -513 for its default, centered siting).
struct Region {
    data: [*const u8; 8],
    linesize: [i32; 8],
    format: av::AVPixelFormat,
    width: i32,
    height: i32,
    chroma_position: (i32, i32),
}

enum Selection {
    Frame { pts: i64, key: i64, exact: bool },
    Timestamp(f64),
}

pub struct Error(pub String, pub bool);
impl Error {
    pub fn into_py(self) -> PyErr {
        if self.1 {
            PyValueError::new_err(self.0)
        } else {
            PyRuntimeError::new_err(self.0)
        }
    }
}
type Result<T> = std::result::Result<T, Error>;
fn failure(message: impl Into<String>) -> Error {
    Error(message.into(), false)
}
fn check(code: i32, operation: &str) -> Result<()> {
    if code >= 0 {
        return Ok(());
    }
    let mut text: [std::ffi::c_char; 256] = [0; 256];
    unsafe {
        av::av_strerror(code, text.as_mut_ptr(), text.len());
    }
    Err(failure(format!(
        "{operation}: {}",
        unsafe { CStr::from_ptr(text.as_ptr()) }.to_string_lossy()
    )))
}
fn name(pointer: *const std::ffi::c_char) -> Option<String> {
    if pointer.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}
pub fn version() -> String {
    name(unsafe { av::av_version_info() }).unwrap_or_default()
}
pub enum Source {
    Path(String),
    Bytes(Vec<u8>),
    File(Py<PyAny>),
}
struct Memory {
    data: Vec<u8>,
    position: usize,
}
enum Reader {
    Bytes(Memory),
    File(Py<PyAny>),
}
unsafe extern "C" fn read_memory(opaque: *mut c_void, buffer: *mut u8, size: i32) -> i32 {
    if size <= 0 {
        return av::AVERROR(libc::EINVAL);
    }
    let reader = &mut *(opaque as *mut Reader);
    let input = match reader {
        Reader::Bytes(input) => input,
        Reader::File(source) => {
            return Python::with_gil(|py| -> PyResult<i32> {
                let data = source.bind(py).call_method1("read", (size,))?;
                let data = data.downcast::<pyo3::types::PyBytes>()?.as_bytes();
                if data.len() > size as usize {
                    return Ok(-5);
                }
                if data.is_empty() {
                    return Ok(av::AVERROR_EOF);
                }
                ptr::copy_nonoverlapping(data.as_ptr(), buffer, data.len());
                Ok(data.len() as i32)
            })
            .unwrap_or(-5)
        }
    };
    let count = (size as usize).min(input.data.len().saturating_sub(input.position));
    if count == 0 {
        return av::AVERROR_EOF;
    }
    ptr::copy_nonoverlapping(input.data.as_ptr().add(input.position), buffer, count);
    input.position += count;
    count as i32
}
unsafe extern "C" fn seek_memory(opaque: *mut c_void, offset: i64, whence: i32) -> i64 {
    let reader = &mut *(opaque as *mut Reader);
    let input = match reader {
        Reader::Bytes(input) => input,
        Reader::File(source) => {
            return Python::with_gil(|py| -> PyResult<i64> {
                let source = source.bind(py);
                if whence & av::AVSEEK_SIZE != 0 {
                    let position: i64 = source.call_method1("seek", (0, 1))?.extract()?;
                    let size: i64 = source.call_method1("seek", (0, 2))?.extract()?;
                    source.call_method1("seek", (position, 0))?;
                    Ok(size)
                } else {
                    source
                        .call_method1("seek", (offset, whence & !av::AVSEEK_FORCE))?
                        .extract()
                }
            })
            .unwrap_or(-5)
        }
    };
    if whence & av::AVSEEK_SIZE != 0 {
        return input.data.len() as i64;
    }
    let base = match whence & !av::AVSEEK_FORCE {
        0 => 0,
        1 => input.position as i64,
        2 => input.data.len() as i64,
        _ => return i64::from(av::AVERROR(libc::EINVAL)),
    };
    let Some(position) = base.checked_add(offset) else {
        return i64::from(av::AVERROR(libc::EINVAL));
    };
    if position < 0 || position > input.data.len() as i64 {
        return i64::from(av::AVERROR(libc::EINVAL));
    }
    input.position = position as usize;
    position
}

pub struct Decoder {
    format: *mut av::AVFormatContext,
    codec: *mut av::AVCodecContext,
    packet: *mut av::AVPacket,
    frame: *mut av::AVFrame,
    rgb_frame: *mut av::AVFrame,
    scale: *mut av::SwsContext,
    scale_config: Option<[i32; 4]>,
    resizers: Vec<Resizer>,
    rotated: Vec<u8>,
    io: *mut av::AVIOContext,
    memory: Option<Box<Reader>>,
    index: i32,
    time_base: av::AVRational,
    audio: bool,
    video_layout: (i32, i32, av::AVPixelFormat),
    draining: bool,
}
// SAFETY: all pointers are uniquely owned. PyO3's mutable borrow plus the Python
// wrapper's RLock serialize access. Custom AVIO callbacks acquire the GIL before
// accessing Python file objects; native operations release it before calling FFmpeg.
unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            self.resizers.clear();
            av::sws_freeContext(self.scale);
            av::av_frame_free(&mut self.frame);
            av::av_frame_free(&mut self.rgb_frame);
            av::av_packet_free(&mut self.packet);
            av::avcodec_free_context(&mut self.codec);
            av::avformat_close_input(&mut self.format);
            if !self.io.is_null() {
                av::av_free((*self.io).buffer as *mut c_void);
                (*self.io).buffer = ptr::null_mut();
                av::avio_context_free(&mut self.io);
            }
        }
    }
}

impl Decoder {
    pub fn file_source(&self) -> Option<&Py<PyAny>> {
        match self.memory.as_deref() {
            Some(Reader::File(source)) => Some(source),
            _ => None,
        }
    }

    pub fn open(source: Source, audio: bool, index: Option<i32>, threads: i32) -> Result<Self> {
        let mut this = Self {
            format: ptr::null_mut(),
            codec: ptr::null_mut(),
            packet: ptr::null_mut(),
            frame: ptr::null_mut(),
            rgb_frame: ptr::null_mut(),
            scale: ptr::null_mut(),
            scale_config: None,
            resizers: Vec::new(),
            rotated: Vec::new(),
            io: ptr::null_mut(),
            memory: None,
            index: 0,
            time_base: av::AVRational { num: 0, den: 1 },
            audio,
            video_layout: (0, 0, av::AVPixelFormat::AV_PIX_FMT_NONE),
            draining: false,
        };
        unsafe {
            let path = match source {
                Source::Path(path) => {
                    Some(CString::new(path).map_err(|_| Error("path contains NUL".into(), true))?)
                }
                input => {
                    let reader = match input {
                        Source::Bytes(data) => Reader::Bytes(Memory { data, position: 0 }),
                        Source::File(source) => {
                            Python::with_gil(|py| {
                                source.bind(py).call_method1("seek", (0, 0)).map(|_| ())
                            })
                            .map_err(|error| failure(error.to_string()))?;
                            Reader::File(source)
                        }
                        Source::Path(_) => unreachable!(),
                    };
                    this.memory = Some(Box::new(reader));
                    let buffer = av::av_malloc(32768) as *mut u8;
                    if buffer.is_null() {
                        return Err(failure("cannot allocate input buffer"));
                    }
                    let opaque = &mut **this.memory.as_mut().unwrap() as *mut Reader as *mut c_void;
                    this.io = av::avio_alloc_context(
                        buffer,
                        32768,
                        0,
                        opaque,
                        Some(read_memory),
                        None,
                        Some(seek_memory),
                    );
                    if this.io.is_null() {
                        av::av_free(buffer as *mut c_void);
                        return Err(failure("cannot allocate input context"));
                    }
                    this.format = av::avformat_alloc_context();
                    if this.format.is_null() {
                        return Err(failure("cannot allocate format context"));
                    }
                    (*this.format).pb = this.io;
                    (*this.format).flags |= av::AVFMT_FLAG_CUSTOM_IO;
                    None
                }
            };
            check(
                av::avformat_open_input(
                    &mut this.format,
                    path.as_ref().map_or(ptr::null(), |p| p.as_ptr()),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                "open input",
            )?;
            check(
                av::avformat_find_stream_info(this.format, ptr::null_mut()),
                "read stream metadata",
            )?;
            let media_type = if audio {
                av::AVMediaType::AVMEDIA_TYPE_AUDIO
            } else {
                av::AVMediaType::AVMEDIA_TYPE_VIDEO
            };
            this.index = match index {
                Some(i) => i,
                None => {
                    av::av_find_best_stream(this.format, media_type, -1, -1, ptr::null_mut(), 0)
                }
            };
            if this.index < 0 || this.index >= (*this.format).nb_streams as i32 {
                return Err(Error(
                    "no valid stream of the requested media type".into(),
                    true,
                ));
            }
            let stream = this.stream();
            let params = (*stream).codecpar;
            if (*params).codec_type != media_type {
                return Err(Error("stream has the wrong media type".into(), true));
            }
            if !audio {
                this.video_layout = (
                    (*params).width,
                    (*params).height,
                    std::mem::transmute::<i32, av::AVPixelFormat>((*params).format),
                );
            }
            this.time_base = (*stream).time_base;
            if this.time_base.den <= 0 || this.time_base.num <= 0 {
                return Err(failure("invalid stream time base"));
            }
            let codec = av::avcodec_find_decoder((*params).codec_id);
            if codec.is_null() {
                return Err(failure("FFmpeg has no decoder for this codec"));
            }
            this.codec = av::avcodec_alloc_context3(codec);
            if this.codec.is_null() {
                return Err(failure("cannot allocate codec context"));
            }
            check(
                av::avcodec_parameters_to_context(this.codec, params),
                "configure decoder",
            )?;
            (*this.codec).thread_count = threads;
            (*this.codec).pkt_timebase = this.time_base;
            check(
                av::avcodec_open2(this.codec, codec, ptr::null_mut()),
                "open decoder",
            )?;
            this.packet = av::av_packet_alloc();
            this.frame = av::av_frame_alloc();
            this.rgb_frame = av::av_frame_alloc();
            if this.packet.is_null() || this.frame.is_null() || this.rgb_frame.is_null() {
                return Err(failure("cannot allocate decoding buffers"));
            }
        }
        Ok(this)
    }

    unsafe fn stream(&self) -> *mut av::AVStream {
        *(*self.format).streams.add(self.index as usize)
    }
    fn seconds(&self, pts: i64) -> f64 {
        pts as f64 * self.time_base.num as f64 / self.time_base.den as f64
    }
    fn begin_pts(&self) -> i64 {
        let start = unsafe { (*self.stream()).start_time };
        if start == av::AV_NOPTS_VALUE {
            0
        } else {
            start
        }
    }
    fn seek(&mut self, pts: i64) -> Result<()> {
        unsafe {
            check(
                av::av_seek_frame(self.format, self.index, pts, av::AVSEEK_FLAG_BACKWARD),
                "seek",
            )?;
            av::avcodec_flush_buffers(self.codec);
            av::av_packet_unref(self.packet);
            av::av_frame_unref(self.frame);
        }
        self.draining = false;
        Ok(())
    }

    pub fn metadata<'py>(
        &self,
        py: Python<'py>,
        apply_rotation: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let data = PyDict::new(py);
        unsafe {
            let stream = &*self.stream();
            let params = &*stream.codecpar;
            data.set_item("stream_index", self.index)?;
            data.set_item("media_type", if self.audio { "audio" } else { "video" })?;
            data.set_item("time_base_num", self.time_base.num)?;
            data.set_item("time_base_den", self.time_base.den)?;
            data.set_item("codec", name(av::avcodec_get_name(params.codec_id)))?;
            data.set_item("bit_rate", params.bit_rate as f64)?;
            data.set_item(
                "duration_seconds_from_header",
                if stream.duration > 0 {
                    Some(self.seconds(stream.duration))
                } else {
                    None
                },
            )?;
            data.set_item(
                "container_duration",
                if (*self.format).duration > 0 {
                    Some((*self.format).duration as f64 / av::AV_TIME_BASE as f64)
                } else {
                    None
                },
            )?;
            data.set_item(
                "begin_stream_seconds_from_header",
                if stream.start_time == av::AV_NOPTS_VALUE {
                    None
                } else {
                    Some(self.seconds(stream.start_time))
                },
            )?;
            if self.audio {
                data.set_item("sample_rate", params.sample_rate)?;
                data.set_item("num_channels", params.ch_layout.nb_channels)?;
                let format: av::AVSampleFormat = std::mem::transmute(params.format);
                data.set_item("sample_format", name(av::av_get_sample_fmt_name(format)))?;
            } else {
                data.set_item("width", params.width)?;
                data.set_item("height", params.height)?;
                data.set_item(
                    "num_frames_from_header",
                    if stream.nb_frames > 0 {
                        Some(stream.nb_frames)
                    } else {
                        None
                    },
                )?;
                let fps = av::av_q2d(stream.r_frame_rate);
                data.set_item(
                    "average_fps_from_header",
                    if fps > 0. { Some(fps) } else { None },
                )?;
                data.set_item(
                    "pixel_aspect_ratio",
                    (
                        params.sample_aspect_ratio.num,
                        params.sample_aspect_ratio.den,
                    ),
                )?;
                let pixel_format: av::AVPixelFormat = std::mem::transmute(params.format);
                data.set_item("pixel_format", name(av::av_get_pix_fmt_name(pixel_format)))?;
                let descriptor = av::av_pix_fmt_desc_get(pixel_format);
                data.set_item(
                    "bit_depth",
                    if descriptor.is_null() {
                        None
                    } else {
                        Some((*descriptor).comp[0].depth)
                    },
                )?;
                data.set_item(
                    "color_range",
                    if params.color_range == av::AVColorRange::AVCOL_RANGE_UNSPECIFIED {
                        None
                    } else {
                        name(av::av_color_range_name(params.color_range))
                    },
                )?;
                data.set_item(
                    "color_space",
                    if params.color_space == av::AVColorSpace::AVCOL_SPC_UNSPECIFIED {
                        None
                    } else {
                        name(av::av_color_space_name(params.color_space))
                    },
                )?;
                data.set_item(
                    "color_primaries",
                    if params.color_primaries == av::AVColorPrimaries::AVCOL_PRI_UNSPECIFIED {
                        None
                    } else {
                        name(av::av_color_primaries_name(params.color_primaries))
                    },
                )?;
                data.set_item(
                    "color_transfer_characteristic",
                    if params.color_trc == av::AVColorTransferCharacteristic::AVCOL_TRC_UNSPECIFIED
                    {
                        None
                    } else {
                        name(av::av_color_transfer_name(params.color_trc))
                    },
                )?;
                let side_data = av::av_packet_side_data_get(
                    params.coded_side_data,
                    params.nb_coded_side_data,
                    av::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
                );
                let rotation = if apply_rotation && !side_data.is_null() && (*side_data).size >= 36
                {
                    let matrix = (*side_data).data as *const i32;
                    let determinant = *matrix as f64 * *matrix.add(4) as f64
                        - *matrix.add(1) as f64 * *matrix.add(3) as f64;
                    if determinant <= 0. {
                        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                            "reflected or singular display matrices are unsupported",
                        ));
                    }
                    Some(av::av_display_rotation_get(matrix))
                } else {
                    None
                };
                data.set_item("rotation", rotation.filter(|r| r.is_finite() && *r != 0.))?;
            }
        }
        Ok(data)
    }

    pub fn scan(&mut self) -> Result<Vec<(i64, i64, bool)>> {
        self.seek(self.begin_pts())?;
        let mut frames = Vec::new();
        unsafe {
            loop {
                let code = av::av_read_frame(self.format, self.packet);
                if code == av::AVERROR_EOF {
                    break;
                }
                check(code, "scan packets")?;
                let packet = &*self.packet;
                if packet.stream_index == self.index && packet.flags & av::AV_PKT_FLAG_DISCARD == 0
                {
                    let pts = if packet.pts != av::AV_NOPTS_VALUE {
                        packet.pts
                    } else {
                        packet.dts
                    };
                    if pts == av::AV_NOPTS_VALUE {
                        return Err(failure("video packet has neither PTS nor DTS"));
                    }
                    frames.push((
                        pts,
                        packet.duration,
                        packet.flags & av::AV_PKT_FLAG_KEY != 0,
                    ));
                }
                av::av_packet_unref(self.packet);
            }
        }
        frames.sort_by_key(|frame| frame.0);
        self.seek(self.begin_pts())?;
        Ok(frames)
    }

    fn next(&mut self) -> Result<bool> {
        unsafe {
            av::av_frame_unref(self.frame);
            loop {
                let code = av::avcodec_receive_frame(self.codec, self.frame);
                if code >= 0 {
                    return Ok(true);
                }
                if code == av::AVERROR_EOF {
                    return Ok(false);
                }
                if code != av::AVERROR(libc::EAGAIN) {
                    check(code, "receive decoded frame")?;
                }
                if self.draining {
                    return Ok(false);
                }
                loop {
                    let code = av::av_read_frame(self.format, self.packet);
                    if code == av::AVERROR_EOF {
                        self.draining = true;
                        check(
                            av::avcodec_send_packet(self.codec, ptr::null()),
                            "drain decoder",
                        )?;
                        break;
                    }
                    check(code, "read packet")?;
                    if (*self.packet).stream_index == self.index
                        && (self.audio || (*self.packet).flags & av::AV_PKT_FLAG_DISCARD == 0)
                    {
                        let code = av::avcodec_send_packet(self.codec, self.packet);
                        av::av_packet_unref(self.packet);
                        check(code, "send packet")?;
                        break;
                    }
                    av::av_packet_unref(self.packet);
                }
            }
        }
    }

    fn frame_pts(&self) -> Result<i64> {
        let frame = unsafe { &*self.frame };
        let pts = if frame.pts != av::AV_NOPTS_VALUE {
            frame.pts
        } else {
            frame.best_effort_timestamp
        };
        if pts == av::AV_NOPTS_VALUE {
            Err(failure("decoded frame has no timestamp"))
        } else {
            Ok(pts)
        }
    }

    fn seek_timestamp(&mut self, seconds: f64) -> Result<()> {
        let begin = self.begin_pts();
        let target = (seconds / seconds_per_tick(self.time_base)).floor() as i64;
        let mut seek_pts = target;
        let mut backoff = (1. / seconds_per_tick(self.time_base)).ceil().max(1.) as i64;
        loop {
            if self.seek(seek_pts).is_ok()
                && self.next()?
                && self.seconds(self.frame_pts()?) <= seconds
            {
                return Ok(());
            }
            if seek_pts <= begin {
                return Err(failure("no frame at or before requested timestamp"));
            }
            seek_pts = if backoff == i64::MAX {
                begin
            } else {
                target.saturating_sub(backoff).max(begin)
            };
            backoff = backoff.saturating_mul(2);
        }
    }

    fn timestamp_frame(&mut self, seconds: f64, cursor: &mut TimestampCursor) -> Result<()> {
        // A known later keyframe avoids decoding entire gaps between sparse requests.
        let seek_forward = if cursor.initialized && !cursor.eof {
            unsafe {
                let pts = (seconds / seconds_per_tick(self.time_base)).floor() as i64;
                let index =
                    av::av_index_search_timestamp(self.stream(), pts, av::AVSEEK_FLAG_BACKWARD);
                let entry = av::avformat_index_get_entry(self.stream(), index);
                !entry.is_null() && (*entry).timestamp > self.frame_pts()?
            }
        } else {
            false
        };
        if !cursor.initialized || seek_forward {
            self.seek_timestamp(seconds)?;
            unsafe {
                av::av_frame_unref(cursor.frame);
            }
            cursor.initialized = true;
            cursor.eof = false;
        } else if !cursor.eof {
            let next_pts = unsafe {
                if (*cursor.frame).pts != av::AV_NOPTS_VALUE {
                    (*cursor.frame).pts
                } else {
                    (*cursor.frame).best_effort_timestamp
                }
            };
            if self.seconds(next_pts) > seconds {
                return Ok(());
            }
            std::mem::swap(&mut self.frame, &mut cursor.frame);
        }
        while !cursor.eof {
            let previous_pts = self.frame_pts()?;
            unsafe {
                av::av_frame_unref(cursor.frame);
                check(
                    av::av_frame_ref(cursor.frame, self.frame),
                    "retain playback frame",
                )?;
            }
            if !self.next()? {
                std::mem::swap(&mut self.frame, &mut cursor.frame);
                cursor.eof = true;
                break;
            }
            let next_pts = self.frame_pts()?;
            if next_pts <= previous_pts {
                return Err(failure(
                    "timestamp mode requires strictly increasing frame PTS",
                ));
            }
            if self.seconds(next_pts) > seconds {
                std::mem::swap(&mut self.frame, &mut cursor.frame);
                return Ok(());
            }
        }
        let frame = unsafe { &*self.frame };
        let end = self.seconds(self.frame_pts()?.saturating_add(frame.duration));
        if frame.duration <= 0 || seconds >= end {
            return Err(failure(
                "timestamp exceeds final frame duration or duration is unknown",
            ));
        }
        Ok(())
    }

    fn indexed_frame(
        &mut self,
        target: i64,
        key: i64,
        exact: bool,
        active_key: &mut Option<i64>,
    ) -> Result<()> {
        if *active_key != Some(key) {
            self.seek(key)?;
            *active_key = Some(key);
        }
        let mut retried_from_beginning = false;
        loop {
            if !self.next()? {
                return Err(failure(format!("no decoded frame at PTS {target}")));
            }
            let pts = self.frame_pts()?;
            if pts < target {
                continue;
            }
            if exact && pts != target {
                if retried_from_beginning {
                    return Err(failure(format!("no decoded frame at exact PTS {target}")));
                }
                self.seek(self.begin_pts())?;
                *active_key = None;
                retried_from_beginning = true;
                continue;
            }
            // Approximate mode picks the first decoded frame at/after its estimated PTS.
            return Ok(());
        }
    }

    pub fn video(
        &mut self,
        request: VideoRequest,
        dtype: OutputDtype,
        ops: &[Op],
    ) -> Result<Video> {
        if self.audio {
            return Err(failure("cannot decode video from audio stream"));
        }
        if !ops.is_empty() && matches!(dtype, OutputDtype::Native) {
            return Err(Error("transforms require RGB output".into(), true));
        }
        let (length, requests, mut cursor) = match request {
            VideoRequest::Frames { targets, exact } => {
                let length = targets.len();
                let mut grouped: BTreeMap<i64, (i64, Vec<usize>)> = BTreeMap::new();
                for (i, (pts, key)) in targets.into_iter().enumerate() {
                    grouped.entry(pts).or_insert((key, Vec::new())).1.push(i);
                }
                let requests: Vec<_> = grouped
                    .into_iter()
                    .map(|(pts, (key, positions))| {
                        (Selection::Frame { pts, key, exact }, positions)
                    })
                    .collect();
                (length, requests, None)
            }
            VideoRequest::Timestamps(times) => {
                if times.iter().any(|t| !t.is_finite()) || times.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(failure("timestamps must be finite, sorted and unique"));
                }
                let length = times.len();
                let requests = times
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| (Selection::Timestamp(t), vec![i]))
                    .collect();
                (length, requests, Some(TimestampCursor::new()?))
            }
        };
        let native = matches!(dtype, OutputDtype::Native);
        let (width, height, source_format) = self.video_layout;
        let (channels, dtype, big_endian) = if native {
            use av::AVPixelFormat::*;
            match source_format {
                AV_PIX_FMT_GRAY8 => (1, OutputDtype::U8, false),
                AV_PIX_FMT_GRAY12LE | AV_PIX_FMT_GRAY16LE => (1, OutputDtype::U16, false),
                AV_PIX_FMT_GRAY16BE => (1, OutputDtype::U16, true),
                AV_PIX_FMT_RGB24 => (3, OutputDtype::U8, false),
                AV_PIX_FMT_RGBA => (4, OutputDtype::U8, false),
                _ => {
                    return Err(Error(
                        "native output does not support this pixel format".into(),
                        true,
                    ))
                }
            }
        } else {
            (3, dtype, false)
        };
        let high_depth = !matches!(dtype, OutputDtype::U8);
        let (mut width, mut height) = (width, height);
        for op in ops {
            match *op {
                Op::Rotate { turns } => {
                    if turns % 2 == 1 {
                        (width, height) = (height, width);
                    }
                }
                Op::Crop {
                    top,
                    left,
                    height: h,
                    width: w,
                } => {
                    if top < 0
                        || left < 0
                        || h <= 0
                        || w <= 0
                        || top + h > height
                        || left + w > width
                    {
                        return Err(Error("crop exceeds the frame".into(), true));
                    }
                    (width, height) = (w, h);
                }
                Op::Resize {
                    height: h,
                    width: w,
                } => {
                    if h <= 0 || w <= 0 {
                        return Err(Error("resize size must be positive".into(), true));
                    }
                    (width, height) = (w, h);
                }
            }
        }
        let (width, height) = (width as usize, height as usize);
        let count = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(channels))
            .ok_or_else(|| failure("frame is too large"))?;
        let total = count
            .checked_mul(length)
            .ok_or_else(|| failure("batch is too large"))?;
        let mut pixels = vec![
            0u8;
            total
                .checked_mul(if high_depth { 2 } else { 1 })
                .and_then(|n| n.checked_add(64))
                .ok_or_else(|| failure("batch is too large"))?
        ];
        let mut pts = vec![0.; length];
        let mut durations = vec![0.; length];
        let mut active_key = None;
        let pipeline = Pipeline::new(ops, (self.video_layout.0, self.video_layout.1));
        let stride = count * if high_depth { 2 } else { 1 };
        for (selection, positions) in requests {
            match selection {
                Selection::Frame { pts, key, exact } => {
                    self.indexed_frame(pts, key, exact, &mut active_key)?
                }
                Selection::Timestamp(seconds) => {
                    self.timestamp_frame(seconds, cursor.as_mut().unwrap())?
                }
            }
            let decoded_pts = self.frame_pts()?;
            let first = positions[0];
            unsafe {
                let frame = &*self.frame;
                if (frame.width, frame.height) != (self.video_layout.0, self.video_layout.1) {
                    return Err(failure("dynamic frame dimensions are unsupported"));
                }
                let input_format: av::AVPixelFormat = std::mem::transmute(frame.format);
                let format = if high_depth {
                    av::AVPixelFormat::AV_PIX_FMT_RGB48LE
                } else {
                    av::AVPixelFormat::AV_PIX_FMT_RGB24
                };
                let (data, linesize, size, rest, slot) = if native {
                    if input_format != source_format {
                        return Err(Error("pixel format changed within stream".into(), true));
                    }
                    let size = (frame.width, frame.height);
                    (
                        frame.data[0] as *const u8,
                        frame.linesize[0],
                        size,
                        &[][..],
                        0,
                    )
                } else if let Some((region, (h, w))) = pipeline
                    .first
                    .and_then(|(rect, size)| Some((region_planes(frame, rect)?, size)))
                {
                    let (data, linesize) =
                        self.resize(0, &region, (w, h), high_depth, Some(frame))?;
                    (data, linesize, (w, h), &pipeline.after_first[..], 1)
                } else {
                    // Full-size color conversion, then every op on RGB.
                    let config = [
                        frame.width,
                        frame.height,
                        input_format as i32,
                        format as i32,
                    ];
                    if self.scale_config != Some(config) {
                        av::sws_freeContext(self.scale);
                        self.scale = scale_context(
                            (frame.width, frame.height, input_format),
                            (frame.width, frame.height, format),
                            0,
                            (-513, -513),
                        );
                        self.scale_config = Some(config);
                    }
                    if self.scale.is_null() {
                        return Err(failure("cannot initialize color conversion"));
                    }
                    configure_colors(self.scale, frame)?;
                    ensure_frame(self.rgb_frame, frame.width, frame.height, format)?;
                    let rows = av::sws_scale(
                        self.scale,
                        frame.data.as_ptr() as *const *const u8,
                        frame.linesize.as_ptr(),
                        0,
                        frame.height,
                        (*self.rgb_frame).data.as_ptr(),
                        (*self.rgb_frame).linesize.as_ptr(),
                    );
                    if rows != frame.height {
                        return Err(failure("color conversion failed"));
                    }
                    let rgb = &*self.rgb_frame;
                    let size = (frame.width, frame.height);
                    (
                        rgb.data[0] as *const u8,
                        rgb.linesize[0],
                        size,
                        &pipeline.ops[..],
                        0,
                    )
                };
                let (data, linesize) =
                    self.transform((data, linesize), size, format, rest, slot)?;
                let row_bytes = width * channels * if high_depth { 2 } else { 1 };
                if data.is_null() || (linesize.unsigned_abs() as usize) < row_bytes {
                    return Err(failure("invalid decoded frame stride"));
                }
                for row in 0..height {
                    ptr::copy_nonoverlapping(
                        data.offset(row as isize * linesize as isize),
                        pixels.as_mut_ptr().add(first * stride + row * row_bytes),
                        row_bytes,
                    );
                }
                for &position in &positions {
                    if position != first {
                        pixels.copy_within(first * stride..(first + 1) * stride, position * stride);
                    }
                    pts[position] = self.seconds(decoded_pts);
                    durations[position] = self.seconds(frame.duration);
                }
            }
        }
        pixels.truncate(total * if high_depth { 2 } else { 1 });
        let pixels = match dtype {
            OutputDtype::F32 => Pixels::F32(
                pixels
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| u16::from_le_bytes([p[0], p[1]]) as f32 / 65535.)
                    .collect(),
            ),
            OutputDtype::U16 => Pixels::U16(
                pixels
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| {
                        if big_endian {
                            u16::from_be_bytes(*p)
                        } else {
                            u16::from_le_bytes(*p)
                        }
                    })
                    .collect(),
            ),
            OutputDtype::U8 => Pixels::U8(pixels),
            OutputDtype::Native => unreachable!(),
        };
        Ok(Video {
            pixels,
            pts,
            durations,
            width,
            height,
            channels,
        })
    }

    /// Applies `ops` to an RGB image (first row, stride) of `size` (width, height): a crop moves the view, a
    /// resize scales it into resize slot `slot` onwards. Returns the result's first row and stride.
    unsafe fn transform(
        &mut self,
        (mut data, mut linesize): (*const u8, i32),
        (mut width, mut height): (i32, i32),
        format: av::AVPixelFormat,
        ops: &[Op],
        mut slot: usize,
    ) -> Result<(*const u8, i32)> {
        let high_depth = format == av::AVPixelFormat::AV_PIX_FMT_RGB48LE;
        let pixel_bytes: isize = if high_depth { 6 } else { 3 };
        for op in ops {
            match *op {
                Op::Rotate { turns } => {
                    // As NumPy's rot90 over (height, width): `turns` counterclockwise quarter turns.
                    let (w, h) = if turns % 2 == 1 {
                        (height, width)
                    } else {
                        (width, height)
                    };
                    // Aligned rows and trailing padding: swscale may read past a row's end.
                    let row = (w as usize * pixel_bytes as usize).next_multiple_of(64);
                    self.rotated.resize(
                        row * h as usize + av::AV_INPUT_BUFFER_PADDING_SIZE as usize,
                        0,
                    );
                    for i in 0..h as isize {
                        for j in 0..w as isize {
                            let (y, x) = match turns {
                                1 => (j, width as isize - 1 - i),
                                2 => (height as isize - 1 - i, width as isize - 1 - j),
                                3 => (height as isize - 1 - j, i),
                                _ => (i, j),
                            };
                            ptr::copy_nonoverlapping(
                                data.offset(y * linesize as isize + x * pixel_bytes),
                                self.rotated
                                    .as_mut_ptr()
                                    .offset(i * row as isize + j * pixel_bytes),
                                pixel_bytes as usize,
                            );
                        }
                    }
                    data = self.rotated.as_ptr();
                    linesize = row as i32;
                    (width, height) = (w, h);
                }
                Op::Crop {
                    top,
                    left,
                    height: h,
                    width: w,
                } => {
                    data =
                        data.offset(top as isize * linesize as isize + left as isize * pixel_bytes);
                    (width, height) = (w, h);
                }
                Op::Resize {
                    height: h,
                    width: w,
                } => {
                    let mut planes = [ptr::null(); 8];
                    let mut strides = [0; 8];
                    (planes[0], strides[0]) = (data, linesize);
                    let region = Region {
                        data: planes,
                        linesize: strides,
                        format,
                        width,
                        height,
                        chroma_position: (-513, -513),
                    };
                    (data, linesize) = self.resize(slot, &region, (w, h), high_depth, None)?;
                    (width, height) = (w, h);
                    slot += 1;
                }
            }
        }
        Ok((data, linesize))
    }

    /// Resizes `region` to `size` (width, height) RGB in resize slot `slot`; `colors` is the decoded frame
    /// whose color space and range apply when the region holds its planes. Returns the first row and stride.
    unsafe fn resize(
        &mut self,
        slot: usize,
        region: &Region,
        (width, height): (i32, i32),
        high_depth: bool,
        colors: Option<&av::AVFrame>,
    ) -> Result<(*const u8, i32)> {
        while self.resizers.len() <= slot {
            self.resizers.push(Resizer::new()?);
        }
        let resizer = &mut self.resizers[slot];
        let (planar, packed, bytes) = if high_depth {
            (
                av::AVPixelFormat::AV_PIX_FMT_GBRP16LE,
                av::AVPixelFormat::AV_PIX_FMT_RGB48LE,
                2,
            )
        } else {
            (
                av::AVPixelFormat::AV_PIX_FMT_GBRP,
                av::AVPixelFormat::AV_PIX_FMT_RGB24,
                1,
            )
        };
        let config = [
            region.width,
            region.height,
            region.format as i32,
            width,
            height,
            planar as i32,
            region.chroma_position.0,
            region.chroma_position.1,
        ];
        if resizer.scale.is_null() || resizer.config != config {
            av::sws_freeContext(resizer.scale);
            // Full-resolution chroma on both sides: interpolated from subsampled planes, not decimated from RGB.
            resizer.scale = scale_context(
                (region.width, region.height, region.format),
                (width, height, planar),
                av::SWS_BILINEAR | av::SWS_FULL_CHR_H_INT | av::SWS_FULL_CHR_H_INP,
                region.chroma_position,
            );
            if resizer.scale.is_null() {
                return Err(failure("cannot initialize resize"));
            }
            resizer.config = config;
        }
        if let Some(frame) = colors {
            configure_colors(resizer.scale, frame)?;
        }
        ensure_frame(resizer.planar, width, height, planar)?;
        ensure_frame(resizer.frame, width, height, packed)?;
        let rows = av::sws_scale(
            resizer.scale,
            region.data.as_ptr(),
            region.linesize.as_ptr(),
            0,
            region.height,
            (*resizer.planar).data.as_ptr(),
            (*resizer.planar).linesize.as_ptr(),
        );
        if rows != height {
            return Err(failure("resize failed"));
        }
        interleave(&*resizer.planar, &*resizer.frame, bytes);
        Ok(((*resizer.frame).data[0], (*resizer.frame).linesize[0]))
    }

    pub fn audio(&mut self, rate: i32, channels: i32, stop: Option<f64>) -> Result<Audio> {
        if !self.audio {
            return Err(failure("cannot decode audio from video stream"));
        }
        if rate <= 0 || channels <= 0 || channels > 64 {
            return Err(Error(
                "sample_rate and num_channels must be positive (channels <= 64)".into(),
                true,
            ));
        }
        // Include codec preroll before time zero (e.g. AAC priming packets).
        if self
            .seek(self.begin_pts().min(0).saturating_sub(1))
            .is_err()
        {
            self.seek(0)?;
        }
        let mut resampler = Resampler(ptr::null_mut());
        let mut planes = vec![Vec::<f32>::new(); channels as usize];
        let mut first_pts = None;
        while self.next()? {
            let pts = self.seconds(self.frame_pts()?);
            first_pts.get_or_insert(pts);
            unsafe {
                let frame = &*self.frame;
                if resampler.0.is_null() {
                    let mut layout = std::mem::zeroed::<av::AVChannelLayout>();
                    av::av_channel_layout_default(&mut layout, channels);
                    let input_format: av::AVSampleFormat = std::mem::transmute(frame.format);
                    let code = av::swr_alloc_set_opts2(
                        &mut resampler.0,
                        &layout,
                        av::AVSampleFormat::AV_SAMPLE_FMT_FLTP,
                        rate,
                        &frame.ch_layout as *const _ as *mut _,
                        input_format,
                        frame.sample_rate,
                        0,
                        ptr::null_mut(),
                    );
                    av::av_channel_layout_uninit(&mut layout);
                    check(code, "configure audio resampling")?;
                    check(av::swr_init(resampler.0), "initialize audio resampling")?;
                }
                resampler.convert(
                    &mut planes,
                    frame.extended_data as *mut *const u8,
                    frame.nb_samples,
                )?;
            }
            if stop.is_some_and(|end| {
                first_pts.unwrap_or(0.) + planes[0].len() as f64 / rate as f64 >= end
            }) {
                break;
            }
        }
        if !resampler.0.is_null() {
            unsafe {
                resampler.convert(&mut planes, ptr::null_mut(), 0)?;
            }
        }
        let samples = planes[0].len();
        Ok(Audio {
            data: planes.into_iter().flatten().collect(),
            samples,
            pts: first_pts.unwrap_or(self.seconds(self.begin_pts())),
        })
    }
}

struct Resampler(*mut av::SwrContext);
impl Drop for Resampler {
    fn drop(&mut self) {
        unsafe {
            av::swr_free(&mut self.0);
        }
    }
}
impl Resampler {
    unsafe fn convert(
        &mut self,
        planes: &mut [Vec<f32>],
        input: *mut *const u8,
        samples: i32,
    ) -> Result<()> {
        let capacity = av::swr_get_out_samples(self.0, samples);
        check(capacity, "get audio output size")?;
        let mut scratch = vec![vec![0f32; capacity as usize + 32]; planes.len()];
        let mut pointers: Vec<_> = scratch
            .iter_mut()
            .map(|p| p.as_mut_ptr() as *mut u8)
            .collect();
        let count = av::swr_convert(self.0, pointers.as_mut_ptr(), capacity, input, samples);
        check(count, "resample audio")?;
        for (plane, output) in planes.iter_mut().zip(scratch) {
            plane.extend_from_slice(&output[..count as usize]);
        }
        Ok(())
    }
}

pub enum Pixels {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}
pub enum OutputDtype {
    Native,
    U8,
    U16,
    F32,
}
pub struct Video {
    pub pixels: Pixels,
    pub pts: Vec<f64>,
    pub durations: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}
pub struct Audio {
    pub data: Vec<f32>,
    pub samples: usize,
    pub pts: f64,
}

/// A region: (top, left, height, width).
type Rect = (i32, i32, i32, i32);

/// How `ops` run on decoded frames. Leading crops and the first resize are one swscale pass from the decoded
/// planes to RGB at the output size, and a display rotation before them moves after that resize (right-angle
/// turns commute with the resampling, up to rounding). Without a resize, frames are converted at full size and
/// the crops select RGB pixels.
struct Pipeline {
    /// Region of the decoded frame and output (height, width) of the first resize, both in the decoded
    /// (unrotated) orientation.
    first: Option<(Rect, (i32, i32))>,
    /// Ops on the first resize's RGB output: the display rotation, then those after the resize.
    after_first: Vec<Op>,
    /// Ops on a frame converted at full size: all of them, for pipelines without a resize and for frames whose
    /// region `region_planes` cannot address.
    ops: Vec<Op>,
}
impl Pipeline {
    /// Plans `ops` for decoded frames of `size` (width, height), dropping resizes to the current size.
    fn new(ops: &[Op], (width, height): (i32, i32)) -> Self {
        let mut size = (height, width);
        let ops: Vec<Op> = ops
            .iter()
            .copied()
            .filter(|op| {
                let previous = size;
                size = match *op {
                    Op::Rotate { turns } if turns % 2 == 1 => (size.1, size.0),
                    Op::Rotate { .. } => size,
                    Op::Crop { height, width, .. } | Op::Resize { height, width } => {
                        (height, width)
                    }
                };
                !matches!(op, Op::Resize { .. }) || size != previous
            })
            .collect();
        let (turns, body) = match ops.first() {
            Some(&Op::Rotate { turns }) => (turns, &ops[1..]),
            _ => (0, &ops[..]),
        };
        let (mut top, mut left) = (0, 0);
        let (mut h, mut w) = if turns % 2 == 1 {
            (width, height)
        } else {
            (height, width)
        };
        for (i, op) in body.iter().enumerate() {
            match *op {
                Op::Crop {
                    top: t,
                    left: l,
                    height: ch,
                    width: cw,
                } => (top, left, h, w) = (top + t, left + l, ch, cw),
                Op::Resize {
                    height: oh,
                    width: ow,
                } => {
                    // Inverse of `transform`'s rotation: the rotated region's pixels in the decoded frame.
                    let region = match turns {
                        1 => (left, width - top - h, w, h),
                        2 => (height - top - h, width - left - w, h, w),
                        3 => (height - left - w, top, w, h),
                        _ => (top, left, h, w),
                    };
                    let output = if turns % 2 == 1 { (ow, oh) } else { (oh, ow) };
                    let mut after_first = Vec::with_capacity(body.len() - i);
                    if turns != 0 {
                        after_first.push(Op::Rotate { turns });
                    }
                    after_first.extend_from_slice(&body[i + 1..]);
                    return Self {
                        first: Some((region, output)),
                        after_first,
                        ops,
                    };
                }
                Op::Rotate { .. } => break,
            }
        }
        Self {
            first: None,
            after_first: Vec::new(),
            ops,
        }
    }
}

/// The planes of `rect` in `frame`. An offset inside a subsampled chroma pair starts the chroma plane at the pair
/// and moves the chroma siting back to match. None when the format cannot be addressed by plane offsets
/// (paletted, bitstream, hardware, or packed with subsampled chroma).
unsafe fn region_planes(frame: &av::AVFrame, rect: Rect) -> Option<Region> {
    let (top, left, height, width) = rect;
    let mut region = Region {
        data: frame.data.map(|p| p as *const u8),
        linesize: frame.linesize,
        format: std::mem::transmute::<i32, av::AVPixelFormat>(frame.format),
        width,
        height,
        chroma_position: (-513, -513),
    };
    if rect == (0, 0, frame.height, frame.width) {
        return Some(region);
    }
    let descriptor = av::av_pix_fmt_desc_get(region.format);
    if descriptor.is_null() {
        return None;
    }
    let descriptor = &*descriptor;
    let flags = descriptor.flags as i32;
    let (shift_x, shift_y) = (
        descriptor.log2_chroma_w as i32,
        descriptor.log2_chroma_h as i32,
    );
    let planar = flags & av::AV_PIX_FMT_FLAG_PLANAR != 0;
    if flags
        & (av::AV_PIX_FMT_FLAG_PAL | av::AV_PIX_FMT_FLAG_BITSTREAM | av::AV_PIX_FMT_FLAG_HWACCEL)
        != 0
        || !planar && (shift_x != 0 || shift_y != 0)
    {
        return None;
    }
    for (plane, pointer) in region.data.iter_mut().enumerate() {
        if pointer.is_null() {
            continue;
        }
        let (index, component) = descriptor.comp[..descriptor.nb_components as usize]
            .iter()
            .enumerate()
            .find(|(_, c)| c.plane as usize == plane)?;
        let chroma = planar && flags & av::AV_PIX_FMT_FLAG_RGB == 0 && (index == 1 || index == 2);
        let (y, x) = if chroma {
            (top >> shift_y, left >> shift_x)
        } else {
            (top, left)
        };
        // `step`: bytes between horizontally adjacent samples of this plane (a whole pixel, padding included,
        // in packed formats).
        *pointer = pointer.offset(
            y as isize * frame.linesize[plane] as isize + x as isize * component.step as isize,
        );
    }
    let position = |shift: i32, offset: i32| {
        let phase = offset & ((1 << shift) - 1);
        if phase == 0 {
            -513
        } else {
            ((128 << shift) - 128) - 256 * phase
        }
    };
    region.chroma_position = (position(shift_x, left), position(shift_y, top));
    Some(region)
}

/// Applies the frame's color space and range to `scale` (unspecified ones keep swscale's defaults).
unsafe fn configure_colors(scale: *mut av::SwsContext, frame: &av::AVFrame) -> Result<()> {
    let mut inverse = ptr::null_mut();
    let mut table = ptr::null_mut();
    let (mut source_range, mut destination_range, mut brightness, mut contrast, mut saturation) =
        (0, 0, 0, 0, 0);
    check(
        av::sws_getColorspaceDetails(
            scale,
            &mut inverse,
            &mut source_range,
            &mut table,
            &mut destination_range,
            &mut brightness,
            &mut contrast,
            &mut saturation,
        ),
        "read color conversion settings",
    )?;
    if frame.color_range != av::AVColorRange::AVCOL_RANGE_UNSPECIFIED {
        source_range = i32::from(frame.color_range == av::AVColorRange::AVCOL_RANGE_JPEG);
    }
    let coefficients = av::sws_getCoefficients(frame.colorspace as i32);
    check(
        av::sws_setColorspaceDetails(
            scale,
            coefficients,
            source_range,
            coefficients,
            destination_range,
            brightness,
            contrast,
            saturation,
        ),
        "configure color conversion",
    )
}

/// (Re)allocates `frame` unless it already has this size and format.
unsafe fn ensure_frame(
    frame: *mut av::AVFrame,
    width: i32,
    height: i32,
    format: av::AVPixelFormat,
) -> Result<()> {
    if (*frame).width != width || (*frame).height != height || (*frame).format != format as i32 {
        av::av_frame_unref(frame);
        (*frame).width = width;
        (*frame).height = height;
        (*frame).format = format as i32;
        check(av::av_frame_get_buffer(frame, 32), "allocate RGB frame")?;
    }
    Ok(())
}

unsafe fn scale_context(
    (source_width, source_height, source_format): (i32, i32, av::AVPixelFormat),
    (width, height, format): (i32, i32, av::AVPixelFormat),
    flags: i32,
    (chroma_x, chroma_y): (i32, i32),
) -> *mut av::SwsContext {
    let context = av::sws_alloc_context();
    if context.is_null() {
        return context;
    }
    let options = [
        (c"srcw", source_width),
        (c"srch", source_height),
        (c"src_format", source_format as i32),
        (c"dstw", width),
        (c"dsth", height),
        (c"dst_format", format as i32),
        (c"sws_flags", flags),
        (c"src_h_chr_pos", chroma_x),
        (c"src_v_chr_pos", chroma_y),
    ];
    for (name, value) in options {
        if av::av_opt_set_int(context as *mut c_void, name.as_ptr(), value.into(), 0) < 0 {
            av::sws_freeContext(context);
            return ptr::null_mut();
        }
    }
    if av::sws_init_context(context, ptr::null_mut(), ptr::null_mut()) < 0 {
        av::sws_freeContext(context);
        return ptr::null_mut();
    }
    context
}

/// Interleaves planar G, B, R (`bytes` per sample) into packed RGB of the same size.
unsafe fn interleave(planar: &av::AVFrame, packed: &av::AVFrame, bytes: usize) {
    let width = planar.width as usize;
    for y in 0..planar.height as isize {
        let plane = |i: usize| {
            std::slice::from_raw_parts(
                planar.data[i].offset(y * planar.linesize[i] as isize),
                width * bytes,
            )
        };
        let (g, b, r) = (plane(0), plane(1), plane(2));
        let out = std::slice::from_raw_parts_mut(
            packed.data[0].offset(y * packed.linesize[0] as isize),
            width * 3 * bytes,
        );
        if bytes == 1 {
            for (((p, &r), &g), &b) in out.as_chunks_mut::<3>().0.iter_mut().zip(r).zip(g).zip(b) {
                *p = [r, g, b];
            }
        } else {
            let (r, g, b) = (
                r.as_chunks::<2>().0,
                g.as_chunks::<2>().0,
                b.as_chunks::<2>().0,
            );
            for (((p, r), g), b) in out.as_chunks_mut::<6>().0.iter_mut().zip(r).zip(g).zip(b) {
                *p = [r[0], r[1], g[0], g[1], b[0], b[1]];
            }
        }
    }
}

fn seconds_per_tick(time_base: av::AVRational) -> f64 {
    time_base.num as f64 / time_base.den as f64
}
