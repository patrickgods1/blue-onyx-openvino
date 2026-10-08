# Blue Onyx OpenVINO

Blue Iris / CodeProject.AI compatible object detection service, written in Rust on native
[OpenVINO](https://github.com/openvinotoolkit/openvino). Runs on Intel integrated and discrete GPUs
(Windows, Linux) and on CPU everywhere (Windows x86_64, Linux x86_64, macOS arm64).

Modeled on [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT) without ONNX Runtime, adding
multi-model serving (`/v1/vision/custom/{model}`), YOLO26 end-to-end models and Intel GPU inference
on Windows.

Status: under construction. See `CLAUDE.md` for the architecture and build commands.
