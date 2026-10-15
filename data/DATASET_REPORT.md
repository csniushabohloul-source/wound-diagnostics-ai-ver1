# گزارش جامع دیتاست‌های پروژه (Comprehensive Project Dataset Report)

این سند مرجع رسمی کلیه دیتاست‌های به‌کاررفته در پروژه تحقیقاتی **«تشخیص‌های مقرون‌به‌صرفه زخم‌های پوستی و مراقبت‌های پیشگیرانه به کمک هوش مصنوعی»** است.

---

## 📊 ماتریس مقایسه‌ای دیتاست‌ها (Dataset Matrix)

| نام دیتاست (Dataset) | محتوا و نمونه‌ها | نقش و کاربرد در پژوهش | مسیر محلی پروژه | لینک‌های رسمی و مرجع |
| :--- | :--- | :--- | :--- | :--- |
| **FUSeg (Foot Ulcer Segmentation)** | ۱۲۱۰ تصویر بالینی از ۸۸۹ بیمار (۱۰۱۰ Train و ۲۰۰ Test) | Train و Test اصلی قطعه‌بندی مرز زخم (Binary Segmentation) | `data/raw/fuseg/` | [GitHub: uwm-bigdata](https://github.com/uwm-bigdata/wound-segmentation) \| [Zenodo: 10.5281/zenodo.4574347](https://doi.org/10.5281/zenodo.4574347) |
| **AZH Wound Database** | حدود ۱۰۱۰ تصویر بیمارستانی؛ همپوشان با Train چلنج FUSeg | همان منبع بالینی FUSeg، جهت آموزش اولیه و استخراج پچ‌ها | `data/raw/azh/` | [UWM Big Data Lab](https://uwm-bigdata.github.io/) \| [GitHub Repo](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/wound_dataset) |
| **Medetec Wound Database** | ۵۹۴ تصویر باکیفیت بالینی از ۱۵ نوع زخم متنوع | آزمون تعمیم‌پذیری (Generalization) به زخم‌های فشاری، وریدی، جراحی و سوختگی | `data/raw/medetec/` | [Medetec Official](http://www.medetec.co.uk/files/medetec-image-databases.html) \| [GitHub Mirror](https://github.com/mlaradji/deep-learning-for-wound-care) |
| **Mendeley Lower Limb & Feet v3** | ۲۶۸۶ تصویر زخم با Mask و ۲۵۷۵ تصویر پای سالم (وضوح ۳۳۱×۳۳۱) | منبع مستقل دوم برای تعمیم و آزمون یادگیری کاذب (Shortcut Learning) | `data/raw/mendeley_lower_limb/` | [Mendeley Data DOI: 10.17632/hsj38fwnvr.3](https://doi.org/10.17632/hsj38fwnvr.3) \| [Kaggle](https://www.kaggle.com/datasets/leoscode/wound-segmentation-images) |
| **LUTSeg** | ۱۴۱ تصویر چندزمانی، ۳۹ بیمار، ۵ بافت، برچسب ۵ متخصص | تفکیک پیکسلی بافت‌ها (Tissue Segmentation)، توافق متخصصان و پایش طولی | `data/raw/lutseg/` | [GitHub: carlosh93/TiSage](https://github.com/carlosh93/TiSage) \| [arXiv:2608.25866](https://arxiv.org/abs/2608.25866) |
| **DFUTissue & WoundTissue** | DFUTissue: ۱۲۶۰ فایل / WoundTissue: ۱۴۷ تصویر با ۶ بافت | بنچمارک چندبافتی و آزمون مستقل تفکیک بافت‌های نکروز، اسلاف و گرانولاسیون | `data/raw/dfutissue/` و `data/raw/wound_tissue/` | [DFUTissueSegNet](https://github.com/uwm-bigdata/DFUTissueSegNet) \| [WoundTissue Repo](https://github.com/akabircs/WoundTissue) |

---

## 📁 ساختار پوشه‌های خام (data/raw Structure)

```text
data/raw/
├── fuseg/                 # ۱۲۱۰ تصویر و ماسک قطعه‌بندی زخم (Train / Val / Test)
├── azh/                   # داده‌های استخراج‌شده مرکز درمان زخم AZH (۲۲۱۸ تصویر)
├── medetec/               # ۵۹۴ تصویر از ۱۵ دسته بالینی زخم
├── mendeley_lower_limb/   # دیتاست مندلی ۲۶۸۶ زخم + ۲۵۷۵ پای سالم (کلاس کنترل)
├── lutseg/                # ۱۴۱ ویزیت طولی با ماسک‌های توافق متخصصین
├── dfutissue/             # ۱۲۶۰ فایل بافتی زخم پای دیابتی (Labeled & Unlabeled)
└── wound_tissue/          # ۱۴۷ تصویر با تفکیک ۶ بافت بالینی
```

---

## 🏷️ کلاس‌های استاندارد بافت زخم (Canonical Tissue Classes)
1. **Background (کلاس ۰):** پوست سالم و پس‌زمینه (شفاف)
2. **Epithelial tissue (کلاس ۱):** بافت اپیتلیال پوششی (سبز)
3. **Slough (کلاس ۲):** اسلاف و فیبرین غیرحیاتی (زرد)
4. **Granulation tissue (کلاس ۳):** گرانولاسیون و بستر التیام‌بخش عروقی (قرمز/صورتی)
5. **Necrotic tissue (کلاس ۴):** بافت مرده و اسکار نکروتیک (سیاه/نارنجی تیره)
6. **Other (کلاس ۵):** سایر ناهنجاری‌ها و مایع اگزودا (فیروزه‌ای)
