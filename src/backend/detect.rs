//! Hardware detection without any runtime library (docs/PLAN.md, "Hardware detection").
//!
//! - **Windows**: DXGI `CreateDXGIFactory1` / `EnumAdapters1`: PCI vendor ID, name, dedicated VRAM;
//!   the software adapter (Microsoft Basic Render Driver) is skipped.
//! - **Linux**: `/sys/class/drm/cardN/device/{vendor,device}` (connector entries such as
//!   `card0-HDMI-A-1` are skipped), names from `/proc/driver/nvidia/gpus/*/information` when the
//!   NVIDIA driver is loaded, VRAM from amdgpu's `mem_info_vram_total`.
//! - **macOS arm64**: one Apple GPU entry (unified memory, so no VRAM figure).
//!
//! The result is computed once per process ([`hardware`]). The parsers are pure functions so they
//! are unit-tested with literal file contents.

use serde::{Serialize, Serializer};
use std::fmt;
use std::sync::OnceLock;

/// PCI vendor IDs.
pub const VENDOR_NVIDIA: u32 = 0x10DE;
pub const VENDOR_INTEL: u32 = 0x8086;
pub const VENDOR_AMD: u32 = 0x1002;
/// Microsoft (the DXGI software/"Basic Render Driver" adapter).
pub const VENDOR_MICROSOFT: u32 = 0x1414;

/// GPU vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Nvidia,
    Intel,
    Amd,
    Apple,
    /// Any other PCI vendor ID.
    Other(u32),
}

impl GpuVendor {
    /// Vendor for a PCI vendor ID.
    pub fn from_pci_id(id: u32) -> Self {
        match id {
            VENDOR_NVIDIA => GpuVendor::Nvidia,
            VENDOR_INTEL => GpuVendor::Intel,
            VENDOR_AMD => GpuVendor::Amd,
            other => GpuVendor::Other(other),
        }
    }
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuVendor::Nvidia => f.write_str("NVIDIA"),
            GpuVendor::Intel => f.write_str("Intel"),
            GpuVendor::Amd => f.write_str("AMD"),
            GpuVendor::Apple => f.write_str("Apple"),
            GpuVendor::Other(id) => write!(f, "vendor 0x{id:04x}"),
        }
    }
}

impl Serialize for GpuVendor {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

/// One hardware GPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuAdapter {
    pub vendor: GpuVendor,
    pub name: String,
    /// Dedicated video memory in MiB; 0 when unknown or unified (iGPU, Apple).
    pub vram_mb: u64,
    /// Platform adapter index: the DXGI adapter index on Windows (what DirectML's `device_id`
    /// means), the DRM card number on Linux, 0 on macOS.
    pub index: u32,
    /// Discrete card (own VRAM) rather than integrated; a heuristic, see [`guess_discrete`].
    pub discrete: bool,
}

/// Detected hardware.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct HardwareInfo {
    /// Hardware GPUs in platform order (see [`GpuAdapter::index`]).
    pub gpus: Vec<GpuAdapter>,
    /// `std::env::consts::OS`: "windows", "linux", "macos", ...
    pub os: String,
    /// `std::env::consts::ARCH`: "x86_64", "aarch64", ...
    pub arch: String,
}

impl HardwareInfo {
    pub fn new(os: &str, arch: &str, gpus: Vec<GpuAdapter>) -> Self {
        Self {
            gpus,
            os: os.to_string(),
            arch: arch.to_string(),
        }
    }

    /// This platform with no GPUs (what the selection assumes when only OpenVINO's device list
    /// is known).
    pub fn this_platform_without_gpus() -> Self {
        Self::new(std::env::consts::OS, std::env::consts::ARCH, Vec::new())
    }

    pub fn is_windows(&self) -> bool {
        self.os == "windows"
    }

    pub fn is_macos(&self) -> bool {
        self.os == "macos"
    }

    /// macOS on Apple silicon (where CoreML is worth picking automatically).
    pub fn is_apple_silicon(&self) -> bool {
        self.is_macos() && self.arch == "aarch64"
    }

    pub fn gpus_of(&self, vendor: GpuVendor) -> impl Iterator<Item = &GpuAdapter> {
        self.gpus.iter().filter(move |g| g.vendor == vendor)
    }

    pub fn has_vendor(&self, vendor: GpuVendor) -> bool {
        self.gpus_of(vendor).next().is_some()
    }
}

impl fmt::Display for GpuAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{} {} ({}", self.index, self.name, self.vendor)?;
        f.write_str(if self.discrete {
            ", discrete"
        } else {
            ", integrated"
        })?;
        if self.vram_mb > 0 {
            write!(f, ", {} MB VRAM", self.vram_mb)?;
        }
        f.write_str(")")
    }
}

/// Detected hardware, computed on first use and cached for the life of the process.
pub fn hardware() -> &'static HardwareInfo {
    static HW: OnceLock<HardwareInfo> = OnceLock::new();
    HW.get_or_init(detect)
}

/// Detect GPUs now (prefer the cached [`hardware`]). Never fails: errors are logged at debug
/// level and yield fewer GPUs.
pub fn detect() -> HardwareInfo {
    let t = std::time::Instant::now();
    let hw = HardwareInfo::new(std::env::consts::OS, std::env::consts::ARCH, detect_gpus());
    tracing::debug!(
        gpus = ?hw.gpus,
        ms = t.elapsed().as_millis() as u64,
        "hardware detection"
    );
    hw
}

#[cfg(windows)]
fn detect_gpus() -> Vec<GpuAdapter> {
    match dxgi::adapters() {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("DXGI adapter enumeration failed: {e:#}");
            Vec::new()
        }
    }
}

#[cfg(target_os = "linux")]
fn detect_gpus() -> Vec<GpuAdapter> {
    linux::adapters(
        std::path::Path::new("/sys/class/drm"),
        std::path::Path::new("/proc/driver/nvidia/gpus"),
    )
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn detect_gpus() -> Vec<GpuAdapter> {
    vec![apple_gpu(&crate::system_info::cpu_name())]
}

#[cfg(not(any(
    windows,
    target_os = "linux",
    all(target_os = "macos", target_arch = "aarch64")
)))]
fn detect_gpus() -> Vec<GpuAdapter> {
    Vec::new()
}

/// The single Apple silicon GPU, named after the chip ("Apple M2 Pro" -> "Apple M2 Pro GPU").
pub fn apple_gpu(chip: &str) -> GpuAdapter {
    let chip = chip.trim();
    let name = if chip.starts_with("Apple") {
        format!("{chip} GPU")
    } else {
        "Apple GPU".to_string()
    };
    GpuAdapter {
        vendor: GpuVendor::Apple,
        name,
        vram_mb: 0,
        index: 0,
        discrete: false,
    }
}

/// Intel discrete GPU PCI device IDs: DG1 (0x4905-0x4909), Alchemist/DG2 Arc A-series and Flex
/// (0x5690-0x56FF), Ponte Vecchio (0x0BD0-0x0BDF), Battlemage Arc B-series (0xE200-0xE2FF).
pub fn intel_device_is_discrete(device_id: u32) -> bool {
    matches!(
        device_id,
        0x4905..=0x4909 | 0x5690..=0x56FF | 0x0BD0..=0x0BDF | 0xE200..=0xE2FF
    )
}

/// Discrete vs integrated without a runtime. NVIDIA is always discrete; Intel by device ID (or a
/// large dedicated VRAM); AMD and others when they report more than 1 GiB of dedicated VRAM (APUs
/// report their small BIOS carve-out, typically 512 MiB); Apple never.
pub fn guess_discrete(vendor: GpuVendor, device_id: u32, vram_mb: u64) -> bool {
    match vendor {
        GpuVendor::Nvidia => true,
        GpuVendor::Apple => false,
        GpuVendor::Intel => intel_device_is_discrete(device_id) || vram_mb > 1024,
        GpuVendor::Amd | GpuVendor::Other(_) => vram_mb > 1024,
    }
}

/// Fallback adapter name when the platform gives none: "Intel GPU [8086:3e92]".
pub fn generic_name(vendor_id: u32, device_id: u32) -> String {
    let vendor = match GpuVendor::from_pci_id(vendor_id) {
        GpuVendor::Other(_) => "Unknown".to_string(),
        v => v.to_string(),
    };
    format!("{vendor} GPU [{vendor_id:04x}:{device_id:04x}]")
}

/// Build an adapter from raw PCI facts. `name` falls back to [`generic_name`].
pub fn adapter_from_ids(
    index: u32,
    vendor_id: u32,
    device_id: u32,
    name: Option<&str>,
    vram_mb: u64,
) -> GpuAdapter {
    let vendor = GpuVendor::from_pci_id(vendor_id);
    let name = name
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| generic_name(vendor_id, device_id));
    GpuAdapter {
        vendor,
        name,
        vram_mb,
        index,
        discrete: guess_discrete(vendor, device_id, vram_mb),
    }
}

/// UTF-16 DXGI description (NUL padded) to a String.
pub fn utf16_name(raw: &[u16]) -> String {
    let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    String::from_utf16_lossy(&raw[..end]).trim().to_string()
}

/// `0x8086\n` (sysfs `vendor`/`device`) or `8086` to a number.
pub fn parse_hex_id(s: &str) -> Option<u32> {
    let s = s.trim();
    let digits = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    if digits.is_empty() {
        return None;
    }
    u32::from_str_radix(digits, 16).ok()
}

/// Card number of a `/sys/class/drm` entry: `card1` -> 1; connectors (`card0-HDMI-A-1`),
/// render nodes (`renderD128`) and anything else -> None.
pub fn drm_card_index(entry: &str) -> Option<u32> {
    let digits = entry.strip_prefix("card")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// `PCI_SLOT_NAME` from a sysfs `uevent` file, lowercased ("0000:01:00.0").
pub fn uevent_pci_slot(uevent: &str) -> Option<String> {
    uevent
        .lines()
        .find_map(|l| l.trim().strip_prefix("PCI_SLOT_NAME="))
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
}

/// What `/proc/driver/nvidia/gpus/<bus>/information` tells us.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvidiaProcInfo {
    /// "NVIDIA GeForce RTX 3060".
    pub model: String,
    /// Lowercased PCI bus location ("0000:01:00.0"), when present.
    pub bus: Option<String>,
}

/// Parse an NVIDIA `information` file (`Key:  value` lines). None without a `Model:` line.
pub fn parse_nvidia_information(text: &str) -> Option<NvidiaProcInfo> {
    let value = |key: &str| {
        text.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            (k.trim() == key).then(|| v.trim().to_string())
        })
    };
    let model = value("Model").filter(|m| !m.is_empty())?;
    let bus = value("Bus Location")
        .map(|b| b.to_ascii_lowercase())
        .filter(|b| !b.is_empty());
    Some(NvidiaProcInfo { model, bus })
}

/// Linux sysfs/procfs scanning. The file system roots are parameters so this is testable on a
/// temporary directory tree.
#[cfg(any(target_os = "linux", test))]
pub(crate) mod linux {
    use super::*;
    use std::path::Path;

    fn read(p: &Path) -> Option<String> {
        std::fs::read_to_string(p).ok()
    }

    /// NVIDIA `information` files under `nvidia_root`, sorted by directory (bus) name.
    fn nvidia_infos(nvidia_root: &Path) -> Vec<(String, NvidiaProcInfo)> {
        let Ok(dir) = std::fs::read_dir(nvidia_root) else {
            return Vec::new();
        };
        let mut v: Vec<(String, NvidiaProcInfo)> = dir
            .flatten()
            .filter_map(|e| {
                let dir_name = e.file_name().to_string_lossy().to_ascii_lowercase();
                let info = parse_nvidia_information(&read(&e.path().join("information"))?)?;
                Some((dir_name, info))
            })
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// GPUs from `drm_root` (`/sys/class/drm`) plus names from `nvidia_root`
    /// (`/proc/driver/nvidia/gpus`). NVIDIA GPUs the driver lists without a DRM card (nvidia-drm
    /// not loaded) are appended after the cards.
    pub fn adapters(drm_root: &Path, nvidia_root: &Path) -> Vec<GpuAdapter> {
        let mut nvidia = nvidia_infos(nvidia_root);
        let mut cards: Vec<(u32, std::path::PathBuf)> = std::fs::read_dir(drm_root)
            .map(|d| {
                d.flatten()
                    .filter_map(|e| {
                        let idx = drm_card_index(&e.file_name().to_string_lossy())?;
                        Some((idx, e.path()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        cards.sort_by_key(|(i, _)| *i);

        let mut out = Vec::new();
        for (idx, path) in cards {
            let dev = path.join("device");
            // Virtual/platform DRM devices (simpledrm, vkms) have no PCI ids.
            let (Some(vendor_id), Some(device_id)) = (
                read(&dev.join("vendor")).as_deref().and_then(parse_hex_id),
                read(&dev.join("device")).as_deref().and_then(parse_hex_id),
            ) else {
                continue;
            };
            let slot = read(&dev.join("uevent"))
                .as_deref()
                .and_then(uevent_pci_slot);
            let mut name = None;
            if vendor_id == VENDOR_NVIDIA {
                let pos = nvidia
                    .iter()
                    .position(|(dir, info)| {
                        slot.as_deref()
                            .is_some_and(|s| dir == s || info.bus.as_deref() == Some(s))
                    })
                    .or(if nvidia.len() == 1 { Some(0) } else { None });
                if let Some(pos) = pos {
                    name = Some(nvidia.remove(pos).1.model);
                }
            }
            // amdgpu reports dedicated VRAM in bytes.
            let vram_mb = read(&dev.join("mem_info_vram_total"))
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|b| b / (1024 * 1024))
                .unwrap_or(0);
            out.push(adapter_from_ids(
                idx,
                vendor_id,
                device_id,
                name.as_deref(),
                vram_mb,
            ));
        }
        let first = out.iter().map(|g| g.index + 1).max().unwrap_or(0);
        for (index, (_, info)) in (first..).zip(nvidia) {
            out.push(adapter_from_ids(
                index,
                VENDOR_NVIDIA,
                0,
                Some(&info.model),
                0,
            ));
        }
        out
    }
}

/// DXGI adapter enumeration (Windows only; the one place in this module with `unsafe`).
#[cfg(windows)]
mod dxgi {
    use super::*;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND, IDXGIFactory1,
    };

    pub fn adapters() -> anyhow::Result<Vec<GpuAdapter>> {
        // SAFETY: plain COM factory creation; the returned interface is reference counted.
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }?;
        let mut out = Vec::new();
        for i in 0u32.. {
            // SAFETY: `factory` is a valid IDXGIFactory1; DXGI_ERROR_NOT_FOUND ends the list.
            let adapter = match unsafe { factory.EnumAdapters1(i) } {
                Ok(a) => a,
                Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(e) => return Err(e.into()),
            };
            // SAFETY: `adapter` is a valid IDXGIAdapter1.
            let desc = match unsafe { adapter.GetDesc1() } {
                Ok(d) => d,
                Err(e) => {
                    tracing::debug!("DXGI adapter {i}: GetDesc1 failed: {e}");
                    continue;
                }
            };
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
                || desc.VendorId == VENDOR_MICROSOFT
            {
                continue;
            }
            let vram_mb = desc.DedicatedVideoMemory as u64 / (1024 * 1024);
            let name = utf16_name(&desc.Description);
            out.push(adapter_from_ids(
                i,
                desc.VendorId,
                desc.DeviceId,
                Some(&name),
                vram_mb,
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn vendor_mapping() {
        assert_eq!(GpuVendor::from_pci_id(0x10DE), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_pci_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_pci_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_pci_id(0x5143), GpuVendor::Other(0x5143));
        assert_eq!(GpuVendor::Other(0x5143).to_string(), "vendor 0x5143");
        assert_eq!(
            serde_json::to_string(&GpuVendor::Nvidia).unwrap(),
            "\"NVIDIA\""
        );
    }

    #[test]
    fn hex_ids_and_card_names() {
        assert_eq!(parse_hex_id("0x8086\n"), Some(0x8086));
        assert_eq!(parse_hex_id("10de"), Some(0x10DE));
        assert_eq!(parse_hex_id("0X1002"), Some(0x1002));
        assert_eq!(parse_hex_id(""), None);
        assert_eq!(parse_hex_id("0x"), None);
        assert_eq!(parse_hex_id("0xzz"), None);

        assert_eq!(drm_card_index("card0"), Some(0));
        assert_eq!(drm_card_index("card12"), Some(12));
        assert_eq!(drm_card_index("card0-HDMI-A-1"), None);
        assert_eq!(drm_card_index("card1-eDP-1"), None);
        assert_eq!(drm_card_index("renderD128"), None);
        assert_eq!(drm_card_index("card"), None);
        assert_eq!(drm_card_index("version"), None);
    }

    #[test]
    fn uevent_and_nvidia_information() {
        let uevent = "DRIVER=nvidia\nPCI_CLASS=30000\nPCI_ID=10DE:2504\n\
                      PCI_SLOT_NAME=0000:01:00.0\nMODALIAS=pci:v000010DEd00002504\n";
        assert_eq!(uevent_pci_slot(uevent).as_deref(), Some("0000:01:00.0"));
        assert_eq!(uevent_pci_slot("DRIVER=i915\n"), None);

        let info = "Model: \t\t NVIDIA GeForce RTX 3060\n\
                    IRQ:   \t\t 140\n\
                    GPU UUID: \t GPU-6a1b2c3d-0000-1111-2222-333344445555\n\
                    Video BIOS: \t 94.06.2f.00.9a\n\
                    Bus Type: \t PCIe\n\
                    DMA Size: \t 47 bits\n\
                    Bus Location: \t 0000:01:00.0\n\
                    Device Minor: \t 0\n";
        assert_eq!(
            parse_nvidia_information(info),
            Some(NvidiaProcInfo {
                model: "NVIDIA GeForce RTX 3060".into(),
                bus: Some("0000:01:00.0".into()),
            })
        );
        assert_eq!(parse_nvidia_information("IRQ: 1\n"), None);
        assert_eq!(parse_nvidia_information("Model:   \n"), None);
    }

    #[test]
    fn discrete_heuristics() {
        // UHD 630 (0x3E92) with its 128 MB DXGI carve-out: integrated.
        assert!(!guess_discrete(GpuVendor::Intel, 0x3E92, 128));
        assert!(guess_discrete(GpuVendor::Intel, 0x56A0, 0)); // Arc A770
        assert!(guess_discrete(GpuVendor::Intel, 0xE20B, 0)); // Arc B580
        assert!(guess_discrete(GpuVendor::Intel, 0x4905, 0)); // DG1
        assert!(!guess_discrete(GpuVendor::Intel, 0x7D55, 0)); // Meteor Lake Arc iGPU
        assert!(guess_discrete(GpuVendor::Intel, 0x1234, 8192));
        assert!(guess_discrete(GpuVendor::Nvidia, 0, 0));
        assert!(guess_discrete(GpuVendor::Amd, 0x73FF, 8176));
        assert!(!guess_discrete(GpuVendor::Amd, 0x1638, 512)); // APU carve-out
        assert!(!guess_discrete(GpuVendor::Apple, 0, 0));
    }

    #[test]
    fn adapter_assembly_and_names() {
        let g = adapter_from_ids(1, 0x10DE, 0x2504, Some(" NVIDIA GeForce RTX 3060 "), 12288);
        assert_eq!(
            g,
            GpuAdapter {
                vendor: GpuVendor::Nvidia,
                name: "NVIDIA GeForce RTX 3060".into(),
                vram_mb: 12288,
                index: 1,
                discrete: true,
            }
        );
        let g = adapter_from_ids(0, 0x8086, 0x3E92, None, 0);
        assert_eq!(g.name, "Intel GPU [8086:3e92]");
        assert!(!g.discrete);
        assert_eq!(generic_name(0x1AF4, 0x1050), "Unknown GPU [1af4:1050]");

        let mut raw = [0u16; 128];
        for (i, c) in "Intel(R) UHD Graphics 630".encode_utf16().enumerate() {
            raw[i] = c;
        }
        assert_eq!(utf16_name(&raw), "Intel(R) UHD Graphics 630");
        assert_eq!(utf16_name(&[]), "");

        assert_eq!(apple_gpu("Apple M2 Pro").name, "Apple M2 Pro GPU");
        assert_eq!(apple_gpu("arm64").name, "Apple GPU");
        assert_eq!(apple_gpu("x").vendor, GpuVendor::Apple);
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn linux_sysfs_tree() {
        let root = std::env::temp_dir().join(format!(
            "bop-detect-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let drm = root.join("drm");
        let nv = root.join("nvidia");
        // card0: Intel iGPU, card1: NVIDIA, card2: virtual (no ids), plus connector/render nodes.
        write(&drm.join("card0/device/vendor"), "0x8086\n");
        write(&drm.join("card0/device/device"), "0x3e92\n");
        write(&drm.join("card0-HDMI-A-1/status"), "connected\n");
        write(&drm.join("renderD128/dev"), "226:128\n");
        write(&drm.join("card1/device/vendor"), "0x10de\n");
        write(&drm.join("card1/device/device"), "0x2504\n");
        write(
            &drm.join("card1/device/uevent"),
            "DRIVER=nvidia\nPCI_SLOT_NAME=0000:01:00.0\n",
        );
        write(
            &drm.join("card2/device/uevent"),
            "DRIVER=simple-framebuffer\n",
        );
        write(&drm.join("card3/device/vendor"), "0x1002\n");
        write(&drm.join("card3/device/device"), "0x73ff\n");
        write(
            &drm.join("card3/device/mem_info_vram_total"),
            "8573157376\n",
        );
        write(
            &nv.join("0000:01:00.0/information"),
            "Model: \t\t NVIDIA GeForce RTX 3060\nBus Location: \t 0000:01:00.0\n",
        );
        // A second NVIDIA GPU without a DRM card.
        write(
            &nv.join("0000:02:00.0/information"),
            "Model: \t\t NVIDIA RTX A2000\nBus Location: \t 0000:02:00.0\n",
        );

        let gpus = linux::adapters(&drm, &nv);
        let summary: Vec<(u32, GpuVendor, &str, u64, bool)> = gpus
            .iter()
            .map(|g| (g.index, g.vendor, g.name.as_str(), g.vram_mb, g.discrete))
            .collect();
        assert_eq!(
            summary,
            vec![
                (0, GpuVendor::Intel, "Intel GPU [8086:3e92]", 0, false),
                (1, GpuVendor::Nvidia, "NVIDIA GeForce RTX 3060", 0, true),
                (3, GpuVendor::Amd, "AMD GPU [1002:73ff]", 8176, true),
                (4, GpuVendor::Nvidia, "NVIDIA RTX A2000", 0, true),
            ]
        );
        // Missing roots: nothing, no panic.
        assert!(linux::adapters(&root.join("none"), &root.join("none")).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn platform_helpers() {
        let mac = HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]);
        assert!(mac.is_macos() && mac.is_apple_silicon() && !mac.is_windows());
        assert!(mac.has_vendor(GpuVendor::Apple) && !mac.has_vendor(GpuVendor::Intel));
        assert!(!HardwareInfo::new("macos", "x86_64", vec![]).is_apple_silicon());
        // Detection on the build host never panics and reports the host platform.
        let hw = hardware();
        assert_eq!(hw.os, std::env::consts::OS);
        assert_eq!(hw.arch, std::env::consts::ARCH);
    }
}
