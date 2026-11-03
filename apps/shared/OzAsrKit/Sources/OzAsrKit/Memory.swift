import Darwin
import Foundation

/// 記憶體量測（MB）。peak = 行程生命週期內的常駐記憶體峰值（getrusage），current = 目前常駐（mach task_info）。
public enum OzMemory {
    public static func peakResidentMB() -> Double {
        var ru = rusage()
        getrusage(RUSAGE_SELF, &ru)
        return Double(ru.ru_maxrss) / 1_048_576  // macOS / iOS 的單位是 bytes
    }

    public static func currentResidentMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.resident_size) / 1_048_576 : 0
    }
}
