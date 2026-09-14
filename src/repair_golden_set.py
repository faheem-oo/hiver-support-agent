import csv

INPUT = "data/processed/golden_set.csv"
OUTPUT = "data/processed/golden_set.csv"
BACKUP = "data/processed/golden_set_before_repair.csv"

# Correct versions of the malformed records.
# Key = golden-set ID.
REPAIRS = {
    113: [
        113, 542388,
        "@AppleSupport @115858 hello, my iphone7plus keeps cutting out every few minutes and giving me this screen. Can you help? https://t.co/1nRQ994LyT",
        "hello, my iphone7plus keeps cutting out every few minutes and giving me this screen. Can you help?",
        "device_performance", "yes", "repeated_device_failure"
    ],

    122: [
        122, 2487237,
        "@AppleSupport and fixing the individual songs by hand. Dreadful solution, but it works. IF you can FIND the songs. Which I can’t, because I can’t see how",
        "and fixing the individual songs by hand. Dreadful solution, but it works. IF you can FIND the songs. Which I can't, because I can't see how",
        "itunes_music", "no", ""
    ],

    124: [
        124, 1748707,
        "@AppleSupport I have the automatic updates on, just checked to be sure and there’s nothing that needs updating",
        "I have the automatic updates on, just checked to be sure and there's nothing that needs updating",
        "ios_update", "no", ""
    ],

    126: [
        126, 1055468,
        "@AppleSupport is it 2 much to ask for my devices to work properly? why am I buying products that are 1k, outdated 1 year &amp; unusable in 5?",
        "is it 2 much to ask for my devices to work properly? why am I buying products that are 1k, outdated 1 year & unusable in 5?",
        "other", "no", ""
    ],

    129: [
        129, 269520,
        "@AppleSupport not sure what happened but after the software update, my iPhone wont send iMessages or connect to internet unless I’m on WiFi",
        "not sure what happened but after the software update, my iPhone wont send iMessages or connect to internet unless I'm on WiFi",
        "connectivity", "no", ""
    ],

    138: [
        138, 1001990,
        "@AppleSupport pushed 11.0.3 update. Battery drains 1% a minute when typing. WiFi issues, WhatsApp working funny, touch non responsive.",
        "pushed 11.0.3 update. Battery drains 1 a minute when typing. WiFi issues, WhatsApp working funny, touch non responsive.",
        "battery_charging", "yes", "multiple_critical_issues"
    ],

    139: [
        139, 989698,
        "@AppleSupport my photos aren’t loading and have a cloud in the corner... tried to delete and they all reappear, so tired of this craziness",
        "my photos aren't loading and have a cloud in the corner... tried to delete and they all reappear, so tired of this craziness",
        "apple_id_icloud", "no", ""
    ],

    151: [
        151, 560473,
        "New update is horrible. I turned my WiFi off and back on and now it won’t load anything, like I don’t have internet @AppleSupport",
        "New update is horrible. I turned my WiFi off and back on and now it won't load anything, like I don't have internet",
        "connectivity", "no", ""
    ],

    152: [
        152, 1716578,
        "@AppleSupport why does this keep happening to my Beats studio wireless? I’ve already had them replaced once for the same issue. https://t.co/4muj15ZLcx",
        "why does this keep happening to my Beats studio wireless? I've already had them replaced once for the same issue.",
        "connectivity", "yes", "repeated_hardware_failure"
    ],

    158: [
        158, 823606,
        "@AppleSupport I did not🙃 Yes, it seems to be working now",
        "I did not Yes, it seems to be working now",
        "other", "no", ""
    ],

    160: [
        160, 1946893,
        "@AppleSupport frustrated ios1103 connections so slow. WiFi useless n cell bad. Pls fix. Please. https://t.co/yn5rQI8k3O",
        "frustrated ios1103 connections so slow. WiFi useless n cell bad. Pls fix. Please.",
        "connectivity", "no", ""
    ],

    164: [
        164, 2583102,
        "@AppleSupport Hope this helps, thanks guys you rock https://t.co/QKg79j48sR",
        "Hope this helps, thanks guys you rock",
        "other", "no", ""
    ],

    166: [
        166, 245959,
        "@AppleSupport yes! https://t.co/zgQICndjFj",
        "yes!",
        "other", "no", ""
    ],

    172: [
        172, 557757,
        "@AppleSupport never mind, there is general #Lag in iOS 11.2 ❗️",
        "never mind, there is general Lag in iOS 11.2",
        "device_performance", "no", ""
    ],

    176: [
        176, 2068308,
        "@AppleSupport The display (original) often does not work anymore, it does not even allow launch applications. You have to do a hard reset",
        "The display original often does not work anymore, it does not even allow launch applications. You have to do a hard reset",
        "screen_display", "yes", "device_screen_failure"
    ],

    193: [
        193, 1639283,
        "@AppleSupport This has nothing to do with my issue.. https://t.co/WDxgOfwTiq",
        "This has nothing to do with my issue..",
        "other", "no", ""
    ],

    195: [
        195, 2903745,
        "@AppleSupport Thanks! Here are screenshots from Storage Info and Finder Info. https://t.co/yEdyQi2Ifz",
        "Thanks! Here are screenshots from Storage Info and Finder Info.",
        "other", "no", ""
    ],

    197: [
        197, 1851596,
        "Hey @115858 and @AppleSupport - does chat even work? It's been 2mins for 30mins now. https://t.co/Vzm4fLd8GB",
        "Hey and - does chat even work? It's been 2mins for 30mins now.",
        "other", "yes", "unresolved_support_issue"
    ],
}


def csv_line(row):
    from io import StringIO

    buffer = StringIO()
    writer = csv.writer(
        buffer,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator=""
    )
    writer.writerow(row)
    return buffer.getvalue()


# Backup first
with open(INPUT, "r", encoding="utf-8", newline="") as f:
    original = f.read()

with open(BACKUP, "w", encoding="utf-8", newline="") as f:
    f.write(original)

lines = original.splitlines()

repaired = []
repaired_ids = set()

for line in lines:
    # Header
    if line.startswith("id,tweet_id,"):
        repaired.append(line)
        continue

    # Every record begins with its numeric ID.
    try:
        record_id = int(line.split(",", 1)[0])
    except ValueError:
        repaired.append(line)
        continue

    if record_id in REPAIRS:
        repaired.append(csv_line(REPAIRS[record_id]))
        repaired_ids.add(record_id)
    else:
        repaired.append(line)

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(repaired) + "\n")

print("Repair complete.")
print("Repaired records:", len(repaired_ids))
print("IDs:", sorted(repaired_ids))
print("Backup:", BACKUP)

missing = set(REPAIRS) - repaired_ids
if missing:
    print("WARNING: These IDs were not found:", sorted(missing))