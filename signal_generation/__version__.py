"""
Version information for Signal Generation system
"""

__version__ = "1.1.0"
__version_date__ = "2025-10-24"
__version_info__ = {
    "version": __version__,
    "date": __version_date__,
    "changes": [
        "v1.1.0 (2025-10-24): رفع مشکل تشخیص کم Doji با معیارهای قابل تنظیم",
        "  - جایگزینی TA-Lib با detector دستی",
        "  - threshold قابل تنظیم برای DojiPattern",
        "  - اسکن همه کندل‌ها در test_pattern.py",
        "  - PatternOrchestrator: پشتیبانی از class و instance"
    ]
}


def get_version_string():
    """نمایش اطلاعات ورژن"""
    return f"Signal Generation v{__version__} ({__version_date__})"


def print_version():
    """چاپ کامل اطلاعات ورژن"""
    print(f"\n{'='*80}")
    print(f"📦 {get_version_string()}")
    print(f"{'='*80}")
    print("تغییرات:")
    for change in __version_info__['changes']:
        print(f"  {change}")
    print(f"{'='*80}\n")
