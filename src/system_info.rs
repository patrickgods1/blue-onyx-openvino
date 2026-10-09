//! Best-effort host information for logging and the status page.

/// CPU brand string.
pub fn cpu_name() -> String {
    cpu_name_impl()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpu_name_impl() -> String {
    let cpuid = raw_cpuid::CpuId::new();
    cpuid
        .get_processor_brand_string()
        .map(|b| b.as_str().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown CPU".to_string())
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn cpu_name_impl() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            && out.status.success()
            && let Ok(s) = String::from_utf8(out.stdout)
            && !s.trim().is_empty()
        {
            return s.trim().to_string();
        }
        return "Apple Silicon".to_string();
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
            for key in ["model name", "Model", "Hardware"] {
                if let Some(v) = text
                    .lines()
                    .find(|l| l.starts_with(key))
                    .and_then(|l| l.split_once(':'))
                    .map(|(_, v)| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                {
                    return v;
                }
            }
        }
    }
    #[allow(unreachable_code)]
    std::env::consts::ARCH.to_string()
}

/// Total physical memory in GiB, if it can be determined.
pub fn total_memory_gb() -> Option<f64> {
    total_memory_bytes().map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0))
}

#[cfg(target_os = "linux")]
fn total_memory_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = text
        .lines()
        .find(|l| l.starts_with("MemTotal:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kb * 1024)
}

#[cfg(target_os = "macos")]
fn total_memory_bytes() -> Option<u64> {
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

#[cfg(windows)]
fn total_memory_bytes() -> Option<u64> {
    // Avoids extra Win32 feature flags: ask CIM through PowerShell, best effort.
    let out = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn total_memory_bytes() -> Option<u64> {
    None
}

/// e.g. "windows x86_64".
pub fn os_description() -> String {
    format!("{} {}", std::env::consts::OS, std::env::consts::ARCH)
}

#[cfg(test)]
mod tests {
    #[test]
    fn nonempty() {
        assert!(!super::cpu_name().is_empty());
        assert!(!super::os_description().is_empty());
        let _ = super::total_memory_gb();
    }
}
