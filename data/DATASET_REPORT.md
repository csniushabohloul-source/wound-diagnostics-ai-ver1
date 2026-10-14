# گزارش فاز ۲: تحلیل و استانداردسازی داده‌ها (Phase 2 Dataset Report)

## ۱. آمار کلی دیتاست پایش طولی (LUTSeg Summary)
- **تعداد کل بیماران ارزیابی‌شده:** 3
- **تعداد کل ارزیابی‌ها / ویزیت‌ها:** 26
- **وضوح هدف پس از پیش‌پردازش تطبیقی:** 256x256 (Letterboxed Padding)
- **کیفیت تصاویر موبایلی:** 16 تصویر استاندارد، 10 نیازمند تنظیم مجدد زاویه/نور

## ۲. توزیع بافت‌های پنج‌گانه زخم (Tissue Class Balance)
| نام بافت | تعداد پیکسل برچسب‌خورده | درصد از کل بستر زخم | رنگ استاندارد |
| :--- | :---: | :---: | :--- |
| Epithelial tissue | 29,832 | 27.00% | RGB |
| Slough | 29,633 | 26.82% | RGB |
| Granulation tissue | 41,932 | 37.95% | RGB |
| Necrotic tissue | 9,106 | 8.24% | RGB |
| Other | 0 | 0.00% | RGB |

## ۳. نمونه نتایج پایش طولی و هشدارهای پیشگیرانه (Preventive Early Warnings)
| کد بیمار | تعداد ویزیت | وضعیت پیشگیرانه | هشدار / توصیه سیستم |
| :--- | :---: | :---: | :--- |
| Patient_10 | 6 | **MODERATE** | MODERATE: Clean wound bed, monitor exudate, re-evaluate in 3-5 days. |
| Patient_20 | 12 | **HIGH** | URGENT: Schedule immediate in-person clinical review, microbial swab, or debridement. |
| Patient_34 | 8 | **HIGH** | URGENT: Schedule immediate in-person clinical review, microbial swab, or debridement. |
