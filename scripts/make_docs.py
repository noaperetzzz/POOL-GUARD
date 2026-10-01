"""Generate a Hebrew Word document describing every Python file in the project.

The signatures, argument names, return annotations and docstrings are extracted
from the source with `ast` rather than typed out by hand, so the document cannot
drift out of step with the code. Re-run it after any change and the description
is correct again.

Usage:
    python scripts/make_docs.py [--out "מסמך.docx"]
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
MUTED = RGBColor(0x60, 0x60, 0x60)
CODE_FONT = "Consolas"


# ---------------------------------------------------------------- descriptions

@dataclass
class FileNote:
    """Hand-written Hebrew commentary for one file."""

    role: str          # one line: what this file is for
    detail: str        # the fuller explanation
    receives: str      # what comes in
    returns: str       # what goes out
    note: str = ""     # a design decision worth knowing


NOTES: dict[str, FileNote] = {
    "poolguard_app.py": FileNote(
        role="נקודת הכניסה של התוכנה. זה הקובץ שמריצים.",
        detail=(
            "קובץ בן שורה אחת שכל תפקידו לייבא את החבילה ולהפעיל אותה. הוא יושב "
            "בשורש הפרויקט ולא בתוך התיקייה pool_guard, וזה לא שרירותי: PyInstaller "
            "מריץ את קובץ הכניסה בשם __main__ בלי חבילת אב, ולכן כל ייבוא יחסי "
            "בתוכו נכשל. קובץ משגר חיצוני שמייבא את החבילה בשמה המלא עוקף את זה."
        ),
        receives="ארגומנטים משורת הפקודה (אופציונלי), מועברים כמו שהם ל-main.",
        returns="קוד יציאה למערכת ההפעלה: 0 בהצלחה, 1 בכשל.",
        note="להריץ: python poolguard_app.py",
    ),
    "pool_guard/main.py": FileNote(
        role="מחליט אם לפתוח חלון או לעבוד משורת פקודה, ותופס קריסות.",
        detail=(
            "בלי ארגומנטים — פותח את החלון. עם ארגומנטים — מניח שמישהו מפעיל "
            "מטרמינל בכוונה ומעביר הכל ל-run. בנוסף הוא עוטף את כל הריצה "
            "ב-try/except: בגרסת EXE ללא קונסולה, חריגה לא מטופלת גורמת לתוכנה "
            "להיעלם בלי שום הודעה — הכשל הגרוע ביותר למי שרק לחץ פעמיים על הקובץ."
        ),
        receives="רשימת ארגומנטים; אם לא הועברה, נלקחת מ-sys.argv.",
        returns="קוד יציאה. במקרה קריסה: כותב poolguard-error.log ליד ה-EXE, מציג "
                "חלונית שגיאה, ומחזיר 1.",
        note="_ensure_console מחבר קונסולה בגרסה חלונית, כדי שהרצה עם דגלים "
             "מטרמינל תדפיס כרגיל.",
    ),
    "pool_guard/__init__.py": FileNote(
        role="מגדיר את החבילה וחושף את המחלקות המרכזיות.",
        detail=(
            "מרכז את השמות שקוד חיצוני צריך — Zone, Tracker, PoolMonitor, "
            "KalmanFilter ועוד — כדי שאפשר יהיה לכתוב from pool_guard import Zone "
            "בלי לדעת באיזה קובץ פנימי הוא יושב. כולל גם את מספר הגרסה ותרשים "
            "הצינור בתיעוד."
        ),
        receives="—",
        returns="—",
    ),
    "pool_guard/paths.py": FileNote(
        role="מוצא קבצים ביחס למקום התוכנה, לא ביחס לתיקיית העבודה.",
        detail=(
            "מפריד בין שני סוגי תיקיות, והבלבול ביניהם שובר את גרסת ה-EXE בצורה "
            "שקשה לשים לב אליה. משאבים לקריאה בלבד (משקולות הגלאי) נוסעים בתוך "
            "הקובץ המהודר ונפרסים לתיקייה זמנית שמחוקה ביציאה. נתונים שנכתבים "
            "(zone.json, poolguard.ini, תיקיית events) חייבים להישמר ליד ה-EXE. "
            "כתיבה בטעות לתיקייה הזמנית נראית כאילו עבדה, ואז נעלמת עם כל "
            "צילומי האזעקות שנשמרו."
        ),
        receives="שמות קבצים או רכיבי נתיב.",
        returns="אובייקטי Path מוחלטים. events_dir יוצר את התיקייה אם אינה קיימת.",
    ),
    "pool_guard/imaging.py": FileNote(
        role="קלט/פלט של תמונות שעובד גם עם נתיבים בעברית, והמרת פריים ל-tkinter.",
        detail=(
            "כאן מטופלות שתי מלכודות של Windows. הראשונה: cv2.imread ו-cv2.imwrite "
            "אינם מסוגלים לטפל בנתיבים שאינם ASCII — הם מעבירים את הנתיב לשכבת ה-C++ "
            "בקידוד המערכת, ולכן קובץ בשם עברי פשוט לא נפתח, והם נכשלים בשקט "
            "ומחזירים None במקום לזרוק שגיאה. כך תמונה תקינה לגמרי נראית כמו קובץ "
            "פגום. הפתרון הוא לקרוא את הבתים בפייתון ולפענח אותם בזיכרון. השנייה: "
            "הצגת מטריצת NumPy בתוך tkinter דורשת בדרך כלל את Pillow; Tk 8.6 קורא "
            "PNG מ-base64 באופן מקורי, כך שקידוד בזיכרון פותר את זה בלי תלות נוספת."
        ),
        receives="נתיבי קבצים, מטריצות תמונה של NumPy, וגדלי תיבה לצורך שינוי גודל.",
        returns="read_image מחזיר מטריצה או None; write_image מחזיר True/False; "
                "open_video מחזיר VideoCapture; fit_scale מחזיר מקדם קנה מידה; "
                "to_photo_image מחזיר tk.PhotoImage.",
        note="את ה-PhotoImage חייבים לשמור בהפניה חיה, אחרת ה-widget מתרוקן בשקט.",
    ),
    "pool_guard/config.py": FileNote(
        role="קורא וכותב את poolguard.ini שיושב ליד התוכנה.",
        detail=(
            "כל מה ששווה לכוון בלי לפתוח קוד נמצא כאן: כמה פריימים רצופים בתוך "
            "האזור לפני התראה, השהיית היציאה, זמן הקירור, סף הגלאי, כל כמה פריימים "
            "לנתח, ומה עושים עם כלב או חתול. הקובץ נכתב עם ברירות המחדל וההערות "
            "בהרצה הראשונה, כך שתמיד יש מה לפתוח ולערוך. מקטע [measurements] "
            "אופציונלי: אם ממולא — מהירויות מדווחות במטרים לשנייה."
        ),
        receives="נתיב לקובץ INI (אופציונלי; ברירת המחדל היא ליד התוכנה).",
        returns="אובייקט Settings. המתודה policy() בונה ממנו Policy עבור המנטר.",
        note="כל קריאה מוגנת בנפרד: ערך שגוי בקובץ ההגדרות לא יעצור את ניטור "
             "הבריכה, אלא ייפול חזרה לברירת המחדל.",
    ),
    "pool_guard/zone.py": FileNote(
        role="הפוליגון המוגן, והבדיקה אם נקודה בתוכו.",
        detail=(
            "האזור נשמר בקואורדינטות מנורמלות (0 עד 1 מרוחב וגובה הפריים) ולא "
            "בפיקסלים, כך שאותו אזור שמור תקף גם אם הווידאו מנוגן ב-1080p וגם אם "
            "הוקטן ל-640 פיקסלים לצורך מהירות. בדיקת נקודה-בתוך-פוליגון ממומשת כאן "
            "ולא מיובאת מ-Shapely: שתים עשרה שורות של הטלת קרן, זה מוריד תלות, "
            "וזה מטפל גם בצורות קעורות — מה שחשוב כי מרפסת בריכה אמיתית היא לעתים "
            "רחוקות מלבן נקי, ובדיקת מלבן חוסם היתה מאשרת נקודות שאינן בפנים."
        ),
        receives="קואורדינטות פיקסל וצורת הפריים, או קובץ JSON לטעינה.",
        returns="point_in_polygon מחזיר True/False; contains_pixel מחזיר True/False; "
                "pixel_points מחזיר מטריצת נקודות בפיקסלים; draw מצייר על הפריים.",
    ),
    "pool_guard/kalman.py": FileNote(
        role="מסנן קלמן לינארי, כתוב מאפס על NumPy.",
        detail=(
            "המסנן כללי ואינו יודע דבר על וידאו או תיבות; המחלקה BoxTrackerFilter "
            "מחברת אותו לעבודה הקונקרטית של החלקת תיבה רועשת לאורך זמן. שלוש סיבות "
            "להשתמש בו בכלל, וכולן קריטיות להיגיון האזעקה: התיבה שהגלאי מחזיר "
            "רועדת בכמה פיקסלים גם על אדם שעומד בלי לזוז; הגלאי מפספס פריימים, "
            "ואדם שנעלם לרגע מאחורי כיסא היה מאבד את זהותו; ווקטור המצב נושא "
            "מהירות, שלא נמדדת ישירות ונופלת מהמסנן בחינם.\n\n"
            "וקטור המצב הוא [cx, cy, w, h, vx, vy] והמדידה היא [cx, cy, w, h]. "
            "כל הגדלים בפיקסלים ובשניות. עדכון השונות משתמש בצורת Joseph, שזהה "
            "אלגברית ל-(I−KH)P אבל נשאר סימטרי וחיובי-מוגדר לאורך אלפי עדכונים, "
            "בעוד הצורה הקצרה עלולה לסחוף ולהתבדר — וזה משנה בתהליך שרץ ימים."
        ),
        receives="תיבה ראשונית (x1,y1,x2,y2), dt בשניות, ופרמטרי רעש: תאוצה "
                 "בפיקסל/שנייה², שינוי גודל בפיקסל/שנייה, ורעש מדידה בפיקסלים.",
        returns="predict מחזיר את התיבה החזויה; update מחזיר את התיבה המסוננת; "
                "התכונות box, velocity ו-speed מחזירות את ההערכה הנוכחית; "
                "gate_distance מחזיר מרחק מהלנוביס ביחידות סיגמה.",
        note="predict מקבל dt לכל צעד בנפרד. הגלאי לא רץ פעם בכל פריים, ולכן "
             "המרווח בין עדכונים אינו מרווח הפריימים; מסנן שמקבל זמן שגוי מצפה "
             "לתנועה שלא היה לה זמן לקרות.",
    ),
    "pool_guard/tracker.py": FileNote(
        role="הופך זיהויים בודדים לכל פריים למסלולים מתמשכים.",
        detail=(
            "הגלאי עונה על השאלה 'מה יש בפריים הזה'. העוקב עונה על השאלה 'האם זה "
            "אותו אדם שראיתי לפני שנייה', וזו השאלה שהיגיון האזעקה באמת צריך. זו "
            "גרסה מצומצמת של SORT: מסנן קלמן לכל מסלול, שיוך לפי חפיפת תיבות "
            "(IoU), ומכונת מצבים קטנה שמאשרת מסלול רק אחרי שנראה מספיק פעמים.\n\n"
            "תוספת אחת מעל SORT הרגיל חשובה כאן במיוחד: כל מסלול שומר מנייה של כל "
            "התוויות שהגלאי אי פעם נתן לו, ומדווח את המנצחת בהצבעה. פריים בודד "
            "יכול בהחלט לתייג גלגל ים צהוב כ'אדם' בביטחון 0.55; ארבעים פריימים "
            "רצופים לא. ההחלטה לפי ההצבעה ולא לפי הפריים האחרון היא מה ששומר על "
            "הסירנה בשקט."
        ),
        receives="רשימת Detection (תיבה, ציון, תווית) ו-dt אמיתי בשניות.",
        returns="רשימת המסלולים המאושרים לאחר העדכון. לכל מסלול: box, label, "
                "label_confidence, detection_rate, speed, travelled, ו-anchor.",
        note="detection_rate מפריד בין שני סוגי כשל: פערים קצרים ומפוזרים הם "
             "סימן לזיהוי לא אמין ונספרים לחובה, בעוד פער אחד ארוך הוא הסתרה "
             "אמיתית ומוחרג. חישוב זהה לשניהם הוריד סירנה לצפצוף על אדם שנעלם "
             "לרגע — בדיוק הכיוון ההפוך במערכת בטיחות.",
    ),
    "pool_guard/detector.py": FileNote(
        role="זיהוי אובייקטים עם YOLOX-S דרך מודול ה-DNN של OpenCV.",
        detail=(
            "בכוונה בלי PyTorch ובלי GPU: המודל הוא קובץ ONNX במשקל 35 מגה־בייט "
            "ו-cv2.dnn מריץ אותו על המעבד, כך שכל הפרויקט מותקן משתי שורות "
            "דרישות ועובד על כל מחשב נייד.\n\n"
            "הבחירה בגלאי שאומן על COCO היא מה שהופך את השאלה 'אדם, כלב, או מזרן "
            "שנשכח' לשאלה שאפשר בכלל לענות עליה: ל-COCO יש מחלקות נפרדות person, "
            "dog ו-cat, ואין לו מחלקה למזרן מתנפח או לכיסא גן — ולכן גלגל ים "
            "בדרך כלל לא מדווח כלל. זה אות חזק לאין שיעור מכל ניתוח תנועה, שיכול "
            "רק לומר 'פיקסלים השתנו' וחסר סיכוי סביב מים וצללים נעים.\n\n"
            "letterbox משנה גודל תוך שמירת יחס הצדדים וממלא את השאר: דחיסת וידאו "
            "טלפון ביחס 16:9 לריבוע היתה מעוותת כל נושא בתמונה."
        ),
        receives="פריים BGR של NumPy. בבנייה: נתיב למודל, סף ביטחון, סף NMS, "
                 "ורשימת המחלקות לשמור.",
        returns="רשימת Detection. כל אחד עם תיבה בקואורדינטות הפריים המקורי, "
                "ציון, ותווית COCO.",
        note="YOLOX הוא anchor-free: כל שורת פלט מכילה היסט מהתא שלה ביחידות "
             "ה-stride שלו, ולכן הפענוח דורש הוספת אינדקס התא והכפלה ב-stride. "
             "רשת התאים קבועה לגודל קלט נתון ונבנית פעם אחת ולא בכל פריים.",
    ),
    "pool_guard/monitor.py": FileNote(
        role="שכבת ההחלטה: מתי מה שנמצא באזור באמת מצדיק אזעקה.",
        detail=(
            "כל מה שלפני זה הוא מדידה — תיבות, מסלולים, תוויות. כאן יושבת המדיניות, "
            "וכאן נמצאת ההנדסה האמיתית של הפרויקט, כי החלק הקשה באזעקת בטיחות אינו "
            "לזהות אנשים. הוא לא לצרוח על החתול של השכן, על צל שנע ברוח, או על "
            "גלגל הים שצף שם מיום שלישי.\n\n"
            "שני רעיונות עושים את רוב העבודה. הראשון: ודאות מדורגת ולא בינארית — "
            "נושא שנתמך בחלשה מעלה NOTICE, צפצוף רך ותמונה שמורה, 'בוא תסתכל', "
            "ורק נושא שנתמך היטב מעלה ALARM. מאחר שהפרדה מושלמת אינה ברת השגה "
            "לשום מערכת ראייה, התכנון אינו מעמיד פנים אחרת: הוא משאיר אדם בלופ "
            "עבור התחום המעורפל. השני: עלויות הכשל אסימטריות — התראת שווא היא "
            "הטרדה, החמצה יכולה להיות ילד שטבע. לכן כל סף כאן מכוון לרגישות, "
            "והרעש שההחלטה הזאת מכניסה מסונן על ידי הסכמה לאורך זמן."
        ),
        receives="רשימת מסלולים מאושרים, צורת הפריים, והזמן הנוכחי כארגומנט.",
        returns="רשימת Event חדשים. כל אירוע: חותמת זמן, מזהה מסלול, תווית, "
                "דרגת חומרה, ביטחון התווית, תיבה, נימוק קריא לאדם, ומהירות "
                "במטרים לשנייה אם האזור מכויל.",
        note="הקובץ לא נוגע בקבצים ולא בקול, ומקבל את הזמן כארגומנט ולא קורא "
             "שעון. לכן ההתנהגות שלו ניתנת לשחזור מלא בטסטים.",
    ),
    "pool_guard/alarm.py": FileNote(
        role="הרמת האזעקה: קול בבית, תמונה בדיסק, שורה בלוג.",
        detail=(
            "הקול מתנגן על חוט רקע. סירנה שחוסמת את הלופ הראשי היתה מקפיאה את "
            "הווידאו לכל אורכה — בדיוק ברגע שבו הכי חשוב להמשיך להסתכל, וגם "
            "עוצרת את הזיהוי בפריימים שהם החשובים ביותר. תבנית האזעקה מחליפה בין "
            "שני צלילים כמו גלאי עשן, בכוונה לא נעימה ונשמעת מחדר אחר; תבנית "
            "ה-NOTICE היא שני צפצופים רכים שאומרים 'תסתכל על המסך', לא 'תרוץ'.\n\n"
            "משלוח לטלגרם לטלפון קיים אך כבוי כברירת מחדל: הוא דורש אסימון בוט "
            "והופך כלי מקומי לכלי שמדבר עם האינטרנט, וזו צריכה להיות בחירה "
            "מפורשת."
        ),
        receives="אובייקט Event והפריים שגרם לו.",
        returns="נתיב לצילום שנשמר, או None אם הכתיבה נכשלה. בנוסף מוסיף שורת "
                "JSON אחת ל-events/log.jsonl ומנגן את תבנית הקול המתאימה.",
        note="הכתיבה עוברת דרך imaging.write_image ולא cv2.imwrite, כי ה-EXE "
             "עלול לשבת בתיקייה עם שם עברי, ושם cv2.imwrite נכשל בשקט.",
    ),
    "pool_guard/layout.py": FileNote(
        role="הופך מדידות סרט מטר של אזור בעל ארבע צלעות לתרשים קרקע שטוח.",
        detail=(
            "האזור הוא מרובע A-B-C-D בסדר ההיקף. הוא אינו חייב להיות מלבן — כל "
            "צורה בעלת ארבע צלעות עובדת, וזה מה שגן או מרפסת בריכה אמיתיים הם "
            "בדרך כלל.\n\n"
            "למה נדרש אלכסון: משולש הוא קשיח, שלוש צלעות קובעות אותו לחלוטין. "
            "מרובע אינו קשיח. בהינתן AB, BC, CD ו-DA בלבד הוא יכול להתקפל כמו "
            "מסגרת על צירים, ולכן ארבעת המספרים האלה מתארים אינסוף צורות שונות. "
            "מדידת אלכסון אחד מפצלת את המרובע לשני משולשים, ושני משולשים קשיחים "
            "ה�קה ל-90 ס\"מ."
        ),
        receives="ארבע צלעות ולפחות אלכסון אחד, בסנטימטרים.",
        returns="אובייקט QuadPlan: ארבע נקודות במערכת קנונית (A בראשית, B על ציר "
                "ה-X החיובי), שגיאת האלכסון העודף, איזה אלכסון קבע את הצורה, "
                "שטח במטרים רבועים, ואורכי הצלעות כפי שהשוחזרו.",
        note="זורק ValueError אם לא נמסר אלכסון, אם אורך אינו חיובי, או אם "
             "האורכים מתארים צורה שאינה יכולה להתקיים במישור.",
    ),
    "pool_guard/calibration.py": FileNote(
        role="ממפה פיקסלים לסנטימטרים על הקרקע באמצעות ארבע נקודות מסומנות.",
        detail=(
            "כשארבע נקודות הגבול ידועות גם בתמונה (בלחיצה) וגם על הקרקע (מ-"
            "layout.solve_quad), השתיים קשורות בהומוגרפיה: הטרנספורמציה "
            "הפרוייקטיבית שמצלמה מיישמת על כל משטח שטוח. ארבעה זוגות נקודות "
            "קובעים אותה בדיוק, וזו הסיבה שמרובע הוא הדבר הטבעי למדוד.\n\n"
            "מה זה מודד במדויק: איפה נושא עומד בסנטימטרים, מה המרחק בין שני "
            "נושאים, ובאיזו מהירות מישהו נע — במטרים לשנייה ולא בפיקסלים לפריים. "
            "כל אלה על מישור הקרקע, וההומוגרפיה מתארת מישור בדיוק.\n\n"
            "מה זה בכוונה אינו מודד: גובה. גרסה קודמת כן ניסתה, מקנה המידה המקומי "
            "של הקרקע ברגלי הנושא. בבדיקה מול מצלמת גן אמיתית — מותקנת גבוה "
            "ומופנית למטה, כמו שמצלמות כאלה תמיד — היא דיווחה על מבוגר בגובה 186 "
            "ס\"מ כבין 154 ל-204, והמספר השתנה כשהתקרב למצלמה. גרוע מזה, היא "
            "תייגה נערה בגובה 160 כ'מבוגר' וכלבה בגובה 60 כ'בגודל ילד'. "
            "הומוגרפיה מתארת משטח שטוח אחד, וראש של אדם עומד אינו עליו בכלל. "
            "לעשות את זה כראוי דורש את נקודת המגוז האנכית או תנוחת מצלמה מלאה. "
            "עד אז התשובה הכנה היא שאין תשובה: מספר שנראה מוסמך ואינו מסוכן יותר "
            "במערכת בטיחות ממספר חסר, כי הוא מזמין החלטות שהמדידה אינה תומכת בהן."
        ),
        receives="ארבע נקודות תמונה בפיקסלים, ארבע נקודות עולם בסנטימטרים, וגודל "
                 "הפריים שעליו נלחצו הנקודות.",
        returns="to_world מחזיר מיקום בסנטימטרים; ground_distance_cm מחזיר מרחק "
                "אמיתי; cm_per_pixel_at מחזיר קנה מידה מקומי; speed_mps מחזיר "
                "מהירות במטרים לשנייה; from_settings מחזיר Calibration או None.",
        note="scaled_to מייצר עותק תקף לרזולוציה אחרת. המנטר מקטין פריימים "
             "לצורך מהירות, ולכן הפיקסלים שהוא מדווח אינם הפיקסלים שעליהם נלחצו "
             "הנקודות; שכחה לעדכן היתה מזיזה בשקט כל מדידה.",
    ),
    "pool_guard/engine.py": FileNote(
        role="צעד אחד של הצינור: דילוג על פריימים, זיהוי, מעקב, החלטה, ציור.",
        detail=(
            "הלופ הנאיבי — לקרוא כל פריים, לזהות עליו, לצייר, לחזור — מנגן וידאו "
            "של 30 פריימים בשנייה בקצב שהגלאי מסוגל לו, כחמישה פריימים בשנייה "
            "על מעבד. התוצאה היא הילוך איטי. במצלמה חיה זה גרוע מאיטי: פריימים "
            "לא מפוענחים נערמים בבאפר והפיגור גדל בלי גבול, כך שמה שרואים נסחף "
            "יותר ויותר לעבר. במצלמה שאמורה לתפוס מישהו נכנס לבריכה, זה מבטל את "
            "כל העניין.\n\n"
            "הפתרון הוא לדלג. רק כל פריים שישי מפוענח ומנותח, השאר נזרקים בלי "
            "שמסתכלים עליהם. חמישה פריימים מנותחים בשנייה צורכים שלושים פריימי "
            "מקור בשנייה — זמן אמת. התמונה מתעדכנת חמש פעמים בשנייה ולא שלושים, "
            "כך שהתנועה נראית מעט קפיצית, אבל היא עכשווית, וכל פריים על המסך הוא "
            "בדיוק הפריים שממנו חושבו המסגרות שלו.\n\n"
            "החלק האחרון הוא הסיבה שזה עדיף על הרצת הזיהוי בחוט רקע. חוטים היו "
            "נותנים תמונה חלקה של 30 פריימים, אבל המסגרות היו תמיד שייכות לפריים "
            "מזיהוי אחד אחורה, וזה היה עולה מנעולים, תור, ומחלקה שלמה של באגים "
            "שבהם חוט רקע נוגע ב-widget. לופ אחד עם דילוג שומר על הווידאו והגלאי "
            "מסונכרנים מעצם הבנייה."
        ),
        receives="מקור (נתיב וידאו או אינדקס מצלמה), אזור, מדיניות, כיול "
                 "אופציונלי, נתיב מודל, סף ביטחון, רוחב להקטנה, גודל דילוג, "
                 "והאם להשמיע קול.",
        returns="step מחזיר Snapshot — פריים מצויר, מסלולים, אירועים, קצב ניתוח, "
                "מיקום וסך הפריימים — או None בסוף הווידאו או בהשהיה. "
                "next_delay_ms מחזיר כמה מילישניות להמתין לפני הצעד הבא, כדי "
                "לנגן בזמן אמת.",
        note="draw_overlay היא פונקציה חופשית ולא מתודה, כי סקריפט הדמו וסקריפט "
             "האיורים צריכים את אותה תמונה בלי להחזיק מצלמה או גלאי.",
    ),
    "pool_guard/setup_gui.py": FileNote(
        role="החלון: סימון האזור, ואחר כך צפייה בו.",
        detail=(
            "הכל קורה בתוך החלון הזה — גלילה לפריים, לחיצה על הפינות, והווידאו "
            "החי בזמן הניטור. אין חלונות OpenCV נפרדים ואין קונסולה. זה תיקון "
            "מכוון: גרסה קודמת שמה את בחירת הפריים בחלונות OpenCV והסתירה את "
            "החלון הראשי בזמן שהיו פתוחים, כך שהממשק נעלם, מסך שחור ישב מאחורי "
            "הכל, ושום דבר על המסך לא אמר מה לעשות הלאה. עירוב שני ארגזי כלים "
            "של חלונות היה הסיבה.\n\n"
            "פריים הייחוס בא מהווידאו או מהמצלמה שמנוטרים, ורק משם. סימון פינות "
            "על תצלום נפרד מזמין את הטעות היחידה שמשבשת בשקט את כל מה שאחריה: "
            "תמונה שצולמה ממקום קצת אחר מזה שבו המצלמה יושבת כעת, כך שהאזור סטוי "
            "בכמה מעלות ואי אפשר לראות את זה בעין.\n\n"
            "הכפתורים Solve ו-Start מושבתים עד שארבע הפינות קיימות. כפתור שמאפיר "
            "ואז נדלק הוא בעצמו שלב האישור — ברור יותר מלבקש הקשה על מקש, ואי "
            "אפשר לפספס אותו כמו שורת אותיות קטנות."
        ),
        receives="לחיצות ובחירות של המשתמש; הגדרות מ-config; פריימים מ-engine.",
        returns="שומר zone.json, מפעיל את המנוע, ומציג וידאו, מצב ולוג אירועים. "
                "main מחזיר קוד יציאה.",
        note="לופ התצוגה רץ על טיימר after של tkinter, והמנוע אומר מתי הפריים "
             "הבא מגיע. כך הווידאו מנוגן במהירות שבה צולם ולא במהירות שהגלאי "
             "מסוגל לה.",
    ),
    "pool_guard/run.py": FileNote(
        role="גרסת שורת הפקודה של המנטר, עם דגלים.",
        detail=(
            "אותו מנוע בדיוק כמו החלון, אבל מונע מלופ while וקורא הקשות מקלדת "
            "בחלון OpenCV. שימושי להרצות אצווה, לבדיקות, ולהרצה במצב headless "
            "שעובר על קובץ שלם מהר כמה שהגלאי מסוגל ומדפיס רק את האירועים."
        ),
        receives="דגלי שורת פקודה: מקור, אזור, כיול, מודל, סף, רוחב, דילוג, "
                 "האם להתחיל נעול, השתקה, headless, dwell ו-grace.",
        returns="קוד יציאה, ומדפיס כל אירוע עם חותמת זמן ונימוק.",
        note="מקשים: a לנעילה, d לפתיחה, רווח להשהיה, חיצים לדילוג של 5 שניות, "
             "s לשמירת פריים, q ליציאה.",
    ),

    "pool_guard/synthetic.py": FileNote(
        role="סצנה מסונתזת עם אמת יסוד, לדמו ולטסטים.",
        detail=(
            "זה מה שמאפשר להריץ ולבדוק את כל הצינור בלי מצלמה, בלי חומר מצולם, "
            "ובלי קובץ המודל במשקל 35 מגה. זה לא תחליף לגלאי האמיתי — דמות "
            "מצוירת אינה משהו ש-YOLOX יזהה. זה תחליף לחומר המצולם: הסצנה באה עם "
            "אמת יסוד, ולכן ScriptedDetector יכול להזין את העוקב בדיוק בסוג הקלט "
            "הפגום שגלאי אמיתי מייצר — תיבות רועדות, פריימים מפוספסים, ותווית "
            "שגויה מדי פעם — וכך שכבת ההחלטה נבחנת מקצה לקצה, דטרמיניסטית.\n\n"
            "הסצנה הבנויה כתובה סביב שלושת המקרים שבאמת חשובים: מבוגר באזור בזמן "
            "שהבריכה פתוחה חייב להישאר בשקט; מזרן שנשכח במים בזמן שהבריכה נעולה, "
            "שהגלאי מתייג כ'אדם' בפריימים מפוזרים, חייב להישאר בשקט; ואדם שנכנס "
            "בזמן שנעול, עם הסתרה קצרה באמצע, חייב להרים אזעקה, והמסלול חייב "
            "לשרוד את ההסתרה ולא להתפצל לשניים."
        ),
        receives="אינדקס פריים, ואופציונלית רקע מוכן.",
        returns="render מחזיר פריים מצויר; ScriptedDetector.detect מחזיר רשימת "
                "Detection; replay הוא גנרטור שמניב לכל פריים את האינדקס, "
                "המסלולים, האירועים והמנטר.",
        note="replay הוא מקור אמת אחד: גם סרטון הדמו וגם טסטי הקצה-לקצה מריצים "
             "אותו צינור בדיוק, במקום שתי לולאות שעלולות להיפרד.",
    ),
}

SCRIPT_NOTES: dict[str, FileNote] = {
    "build_exe.py": FileNote(
        role="בונה את PoolGuard.exe עם PyInstaller.",
        detail=(
            "משקולות הגלאי נארזות בתוך הקובץ המהודר, כך שהתוצאה היא קובץ בודד "
            "שאפשר להעתיק לכל מקום. ברירת המחדל היא בנייה חלונית ללא קונסולה: "
            "החלון מציג את לוג האירועים, ולכן טרמינל שחור מאחוריו הוא רעש בלבד — "
            "ו-OpenCV כותב לשם אזהרות שנראות מבהילות ואינן אומרות דבר למי שמשתמש "
            "בכלי. הדגל --console קיים לניפוי כשל בהפעלה.\n\n"
            "SciPy מוחרג מהבנייה: הוא משמש רק לשיוך אופטימלי בין זיהויים "
            "למסלולים, ל-tracker יש נפילה חזרה לשיוך חמדני, והשארתו בחוץ חוסכת "
            "כשישים מגה־בייט."
        ),
        receives="--onedir, --name, --console.",
        returns="0 בהצלחה. מדפיס את הנתיב והגודל של הקובץ שנבנה.",
    ),
    "conftest.py": FileNote(
        role="מאפשר ל-pytest למצוא את החבילה מכל מקום בעץ.",
        detail="ארבע שורות שמוסיפות את שורש הפרויקט ל-sys.path.",
        receives="—",
        returns="—",
    ),
    "scripts/download_model.py": FileNote(
        role="מוריד את קובץ המודל ONNX לתיקיית models.",
        detail=(
            "המודל שוקל 35 מגה ולכן אינו נשמר ב-git. הוא מגיע מ-OpenCV Model Zoo, "
            "מיוצא עבור cv2.dnn, ואינו דורש PyTorch או GPU. הסקריפט מוודא את "
            "גודל הקובץ ומוחק אותו אם אינו תואם — קובץ קצר בדרך כלל אומר שנשמר "
            "מצביע LFS או דף שגיאה במקום המודל, וזה היה נכשל בבלבול הרבה יותר "
            "מאוחר."
        ),
        receives="—",
        returns="0 בהצלחה, 1 אם ההורדה נכשלה.",
    ),
    "scripts/demo_synthetic.py": FileNote(
        role="מריץ את הצינור על הסצנה המסונתזת וכותב סרטון מוער.",
        detail="בלי מצלמה, בלי חומר מצולם, ובלי קובץ מודל. זה מה שמייצר את "
               "האנימציה ב-README והדרך המהירה ביותר לראות את היגיון ההחלטה עובד.",
        receives="--out, --gif, --snapshots.",
        returns="0, ומדפיס כל אירוע עם מספר הפריים וסיכום בסוף.",
    ),
    "scripts/check_detector.py": FileNote(
        role="בדיקת שפיות של הגלאי על תמונה בודדת.",
        detail="שווה להריץ פעם אחת אחרי הורדת המודל, וזו הדרך המהירה לראות אם "
               "זווית מצלמה או תנאי אור מקשים על הגלאי לפני שמאשימים את היגיון "
               "האזעקה בהחמצה.",
        receives="נתיב תמונה, --model, --conf, --all-classes, --out.",
        returns="0, ומדפיס כל זיהוי עם תווית, ציון ותיבה.",
    ),
    "scripts/make_figures.py": FileNote(
        role="מייצר מחדש את האיורים של ה-README מהצינור האמיתי.",
        detail=(
            "שני פלטים: גרף המעקב דרך הסתרה, שמראה את המיקום האמיתי, הזיהויים "
            "הרועשים והערכת המסנן; ודף שלושה פריימים מוערים — שקט בזמן שהבריכה "
            "פתוחה, NOTICE על דמות שנראית חלקית, ו-ALARM על מי שנכנס."
        ),
        receives="—",
        returns="0. כותב docs/kalman_occlusion.png ו-docs/stills.jpg.",
        note="השיוך לאמת היסוד לוקח את המסלול הקרוב ביותר ולא את הראשון בטווח. "
             "גרסה קודמת לקחה את הראשון, וברגע שמסלול המזרן נסחף לתוך הטווח הוא "
             "צויר בשקט כפורץ — מה שגרם למסנן להיראות פי שלושה גרוע מהזיהויים "
             "הגולמיים שהוא ניקה.",
    ),
    "scripts/make_docs.py": FileNote(
        role="מייצר את המסמך הזה.",
        detail=(
            "החתימות, שמות הארגומנטים, הערכים המוחזרים והתיעוד נשלפים מהקוד עם "
            "ast ולא מוקלדים ביד, כך שהמסמך אינו יכול להיסחף מהקוד. מריצים אותו "
            "מחדש אחרי שינוי והתיאור נכון שוב."
        ),
        receives="--out.",
        returns="0. כותב את קובץ ה-Word.",
    ),
}

TEST_NOTES: dict[str, str] = {
    "tests/test_kalman.py": "התכונות שכל השאר מסתמך עליהן: התכנסות על מסלול נקי, "
        "שחזור מהירות שאינה נמדדת, גלישה דרך פער בזיהויים, החלקה שמנצחת את "
        "המדידות הגולמיות, אי-ודאות שגדלה בחיזוי וקטנה בעדכון, ושונות שנשארת "
        "סימטרית לאורך שלושת אלפים עדכונים.",
    "tests/test_tracker.py": "זהות מסלול לאורך פריימים, שרידות הסתרה קצרה עם "
        "אותו מזהה, נשירה אחרי היעדרות ארוכה, הצבעת רוב על תווית, וההפרדה בין "
        "ריצוד להסתרה. גם שזיהוי רחוק אינו חוטף מסלול קיים.",
    "tests/test_zone.py": "נקודה בתוך פוליגון עבור ריבוע ועבור צורת L קעורה, "
        "קרן שעוברת בדיוק דרך קודקוד, אי-תלות ברזולוציה, ושמירה וטעינה.",
    "tests/test_monitor.py": "הטסטים החשובים ביותר. באג כאן או מעיר את המשפחה "
        "בשלוש לפנות בוקר בגלל חתול, או שותק בזמן שילד במים. כולל: בריכה פתוחה "
        "שותקת, נעולה מרימה אזעקה, מי שבחוץ אינו פורץ, השהיית יציאה, תפאורה "
        "לפי מיקום, קירור, ומדיניות לפי מחלקה.",
    "tests/test_scene.py": "מקצה לקצה על סצנה עם חמישה נושאים בבת אחת: בדיוק "
        "אזעקה אחת והתראה אחת, שקט בזמן שפתוח, שקט על המזרן, שקט על הכלב שבחוץ, "
        "והאזעקה נורה על מי שנכנס. שינוי סף מיטיב שמחזיר התראת שווא נכשל ב-CI "
        "במקום להתגלות בסירנה בחצות.",
    "tests/test_layout.py": "מלבן שחוזר כמלבן, מרובע לא סדיר שעובר הלוך וחזור, "
        "המדידות האמיתיות של הגן שמתגלות כעקביות, שני האלכסונים שנותנים אותה "
        "צורה, ואלכסון שגוי שמופיע כשגיאה גדולה.",
    "tests/test_calibration.py": "הפינות המסומנות ממופות למיקומן המדוד, נקודה "
        "פנימית עוברת הלוך וחזור, מרחקים על המישור מדויקים, קנה המידה משתנה "
        "לרוחב הפריים, ומהירות מדווחת במטרים לשנייה. וגם: שהערכת גובה אינה "
        "מוצעת — בכוונה, לא בשכחה.",
    "tests/test_imaging.py": "הטסט שנולד מהבאג האמיתי: שמות קבצים בעברית, "
        "ברוסית, עם רווח ועם תווים מוטעמים, שכולם עוברים הלוך וחזור. וגם קובץ "
        "חסר, קובץ ריק וקובץ שאינו תמונה.",
    "tests/test_config.py": "הקובץ נוצר עם ברירות מחדל, כל הגדרה נושאת הערה, "
        "עריכה משנה את המדיניות, וערך שטות נופל חזרה לברירת מחדל במקום להפיל "
        "את התוכנה.",
    "tests/test_engine.py": "דילוג הפריימים עצמו: פריים אחד מכל שישה מנותח, "
        "המיקום מתקדם בגודל הדילוג, קיצוב הזמן משאיר מרווח כשהזיהוי מהיר "
        "ומתייאש כשהוא איטי מדי, השהיה עוצרת את הווידאו, ודילוג מנקה מסלולים.",
}

GROUPS = [
    (
        "הפעלה ותצורה",
        "הקבצים שמתחילים את התוכנה ומחזיקים את ההגדרות והנתיבים.",
        ["poolguard_app.py", "pool_guard/main.py", "pool_guard/__init__.py",
         "pool_guard/paths.py", "pool_guard/config.py", "pool_guard/imaging.py"],
    ),
    (
        "ליבת העיבוד",
        "הצינור עצמו: פריים נכנס, החלטה יוצאת. ארבעה שלבים, כל אחד בקובץ נפרד.",
        ["pool_guard/detector.py", "pool_guard/kalman.py", "pool_guard/tracker.py",
         "pool_guard/zone.py", "pool_guard/monitor.py", "pool_guard/alarm.py",
         "pool_guard/engine.py"],
    ),
    (
        "מדידות עולם אמיתי",
        "אופציונלי. הופך פיקסלים לסנטימטרים ומהירויות למטרים לשנייה.",
        ["pool_guard/layout.py", "pool_guard/calibration.py"],
    ),
    (
        "ממשק משתמש",
        "החלון שבו עובדים.",
        ["pool_guard/setup_gui.py"],
    ),
    (
        "בדיקות וכלי עזר",
        "הסצנה המסונתזת, סקריפטי התחזוקה, והבנייה.",
        ["pool_guard/synthetic.py", "build_exe.py", "conftest.py",
         "scripts/download_model.py", "scripts/demo_synthetic.py",
         "scripts/check_detector.py", "scripts/make_figures.py",
         "scripts/make_docs.py"],
    ),
]


# ------------------------------------------------------------------ extraction

def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a def line the way it appears in the source."""
    parts = []
    args = node.args

    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for argument, default in zip(positional, defaults):
        text = argument.arg
        if argument.annotation is not None:
            text += f": {ast.unparse(argument.annotation)}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        parts.append(text)

    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        text = argument.arg
        if argument.annotation is not None:
            text += f": {ast.unparse(argument.annotation)}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        parts.append(text)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    rendered = f"{node.name}({', '.join(parts)})"
    if node.returns is not None:
        rendered += f" -> {ast.unparse(node.returns)}"
    return rendered


def first_line(node) -> str:
    """The first sentence of a docstring, or an empty string."""
    text = ast.get_docstring(node)
    if not text:
        return ""
    return text.strip().split("\n")[0].strip()


@dataclass
class Member:
    kind: str      # "class" or "function"
    signature: str
    summary: str
    methods: list[tuple[str, str]]


def read_members(path: Path) -> list[Member]:
    """Public classes and functions of one module, in source order."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    members: list[Member] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                (signature(child), first_line(child))
                for child in node.body
                if isinstance(child, ast.FunctionDef) and not child.name.startswith("_")
            ]
            bases = [ast.unparse(base) for base in node.bases]
            heading = node.name + (f"({', '.join(bases)})" if bases else "")
            members.append(Member("class", heading, first_line(node), methods))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            members.append(Member("function", signature(node), first_line(node), []))
    return members


# --------------------------------------------------------------- document build

def set_rtl(paragraph) -> None:
    """Mark a paragraph right-to-left, which python-docx does not expose."""
    properties = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    properties.append(bidi)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def hebrew(document, text: str, size: int = 11, bold: bool = False,
           colour: RGBColor | None = None, space_after: int = 6):
    """A right-to-left Hebrew paragraph."""
    paragraph = document.add_paragraph()
    set_rtl(paragraph)
    paragraph.paragraph_format.space_after = Pt(space_after)
    run = paragraph.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "Arial"
    if colour is not None:
        run.font.color.rgb = colour
    return paragraph


def code_line(document, text: str, indent: int = 0) -> None:
    """A left-to-right monospace line, for signatures."""
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.left_indent = Pt(18 + indent)
    run = paragraph.add_run(text)
    run.font.size = Pt(9)
    run.font.name = CODE_FONT
    run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)


def label_value(document, label: str, value: str) -> None:
    """A bold Hebrew label followed by its explanation, on one paragraph."""
    paragraph = document.add_paragraph()
    set_rtl(paragraph)
    paragraph.paragraph_format.space_after = Pt(4)
    bold_run = paragraph.add_run(f"{label}: ")
    bold_run.font.bold = True
    bold_run.font.size = Pt(10.5)
    bold_run.font.name = "Arial"
    bold_run.font.color.rgb = ACCENT
    run = paragraph.add_run(value)
    run.font.size = Pt(10.5)
    run.font.name = "Arial"


def build(out_path: Path) -> None:
    document = Document()

    section = document.sections[0]
    section.left_margin = section.right_margin = Pt(56)

    style = document.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    # ---- title page
    title = document.add_paragraph()
    set_rtl(title)
    run = title.add_run("Pool Guard")
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = ACCENT
    run.font.name = "Arial"

    hebrew(document, "מדריך קבצי הפייתון של התוכנה", size=15, bold=True)
    hebrew(
        document,
        "מצלמה שמנטרת אזור מוגדר ומרימה אזעקה כשהאזור אמור להיות ריק. "
        "המסמך הזה מפרט כל קובץ פייתון בפרויקט: מה תפקידו, מה הוא מקבל, מה הוא "
        "מחזיר, ואיזה החלטות תכנון עומדות מאחוריו.",
        colour=MUTED,
    )
    hebrew(
        document,
        "החתימות והערכים המוחזרים במסמך זה נשלפים מקוד המקור עצמו ולא הוקלדו ביד, "
        "כדי שהמסמך לא יוכל להיסחף מהקוד. להפקה מחדש: python scripts/make_docs.py",
        size=9, colour=MUTED, space_after=14,
    )

    counted = sum(len(files) for _, _, files in GROUPS) + len(TEST_NOTES)
    hebrew(document, f"סך הכל {counted} קבצי פייתון, מחולקים לחמש קבוצות ולתיקיית בדיקות.",
           size=10, colour=MUTED, space_after=18)

    # ---- groups
    for group_index, (group_name, group_blurb, files) in enumerate(GROUPS, start=1):
        heading = document.add_paragraph()
        set_rtl(heading)
        heading.paragraph_format.space_before = Pt(16)
        run = heading.add_run(f"{group_index}. {group_name}")
        run.font.size = Pt(17)
        run.font.bold = True
        run.font.color.rgb = ACCENT
        run.font.name = "Arial"
        hebrew(document, group_blurb, size=10, colour=MUTED, space_after=10)

        for relative in files:
            path = ROOT / relative
            note = NOTES.get(relative) or SCRIPT_NOTES.get(relative)
            if note is None or not path.exists():
                continue

            name_paragraph = document.add_paragraph()
            name_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            name_paragraph.paragraph_format.space_before = Pt(12)
            name_paragraph.paragraph_format.space_after = Pt(2)
            name_run = name_paragraph.add_run(relative)
            name_run.font.size = Pt(13)
            name_run.font.bold = True
            name_run.font.name = CODE_FONT

            lines = len(path.read_text(encoding="utf-8").splitlines())
            hebrew(document, f"{note.role}   ({lines} שורות)", size=10.5, bold=True,
                   space_after=6)

            for block in note.detail.split("\n\n"):
                hebrew(document, block, size=10.5)

            label_value(document, "מקבל", note.receives)
            label_value(document, "מחזיר", note.returns)
            if note.note:
                label_value(document, "לתשומת לב", note.note)

            members = read_members(path)
            if members:
                hebrew(document, "מה יש בקובץ:", size=10, bold=True, colour=ACCENT,
                       space_after=2)
                for member in members:
                    prefix = "class " if member.kind == "class" else "def "
                    code_line(document, prefix + member.signature)
                    if member.summary:
                        hebrew(document, member.summary, size=9, colour=MUTED,
                               space_after=2)
                    for method_signature, method_summary in member.methods:
                        code_line(document, "." + method_signature, indent=16)
                        if method_summary:
                            hebrew(document, method_summary, size=8.5, colour=MUTED,
                                   space_after=1)

    # ---- tests
    heading = document.add_paragraph()
    set_rtl(heading)
    heading.paragraph_format.space_before = Pt(18)
    run = heading.add_run("6. בדיקות")
    run.font.size = Pt(17)
    run.font.bold = True
    run.font.color.rgb = ACCENT
    run.font.name = "Arial"
    hebrew(
        document,
        "כל הבדיקות רצות בלי מצלמה, בלי חומר מצולם ובלי קובץ המודל, הודות לסצנה "
        "המסונתזת. להרצה: python -m pytest tests -q",
        size=10, colour=MUTED, space_after=10,
    )

    for relative, blurb in TEST_NOTES.items():
        path = ROOT / relative
        if not path.exists():
            continue
        name_paragraph = document.add_paragraph()
        name_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        name_paragraph.paragraph_format.space_before = Pt(8)
        name_paragraph.paragraph_format.space_after = Pt(2)
        run = name_paragraph.add_run(relative)
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.name = CODE_FONT

        tree = ast.parse(path.read_text(encoding="utf-8"))
        count = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        )
        hebrew(document, f"{blurb}   ({count} בדיקות)", size=10)

    document.save(str(out_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "docs" / "POOL-GUARD-קבצים.docx"))
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
