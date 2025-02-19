# Known Errors
## Common Causes
Checked LJ: 1, 3, 4, 11, 29, 60, 77, 104, 105, 114, 122, 130, 159, 165, 115, 183, 200, 235

Checked VR: 13, 167, 27, 84, 50, 184, 108, 221, 177

Notes: 
- 184 picked up two extra meetings from other suffrage societies that were listed next to the "forthcoming events"(minor error IMO).
- 221 picked up additional meetings because the list was very short (1 meeting). The meetings picked up are mostly relevant (although some come from other societies), and come from free text descriptives of meetings from the "News of Societies and Federations". Still OK in my book.
- 177 has a typo (OCR error, OK): "Orieff" in csv should be "Crieff"

### 122
- Very weird and persistant "RECITATION" error by Gemini (meaning that it is outputting bits of its training dataset and it is forbidden)
- Needed to run Gemini on the web UI "by hand"

### 200
Also picks up the table of "Coming events" which might also be relevant.

## Suffragette
Checked: 1, 13, 26, 31, 32, 36, 39, 44

### 31 (to drop?)
- One false positive, "flower fair and festival" is not a political meeting (there is no forthcoming meetings table)

### 33 (to drop?)
- 2 false positives, "garden fair" and "summer fair" are not political meetings (there is no forthcoming meetings table)

### 36
- One false positive: flower festival

### 44
- One false positive: Worthing

### To check again from now on

### 71
- Missed events: Chelsea, Chiswick, Harlesden
- Row 25 speaker continues to 26
- Row 28 missing speaker name
- Row 32 "miss pankerd" error
- Row 46 missing speaker
- Row 47 missed Croydon Katherine St event
- Row 55 failed to correct "Dindee" to "Dundee"
- Row 68 missed events: Bexhill, Cardiff, Leicester
- Rows 68-69 wrong speaker
- Row 71 speaker in 72
- Row 74 incomplete speakers
- Row 76 missing street address
- Row 78 street number is 27

### 83
- First two rows contain extra sentences with invalid speakers
- Row 35 date off by -1

### 85 (to drop?)
- No forthcoming meetings table in the issue

### 86 (to drop?)
- No forthcoming meetings table in the issue

### 87 (to drop?)
- No forthcoming meetings table in the issue

### 87
- No programme of events table
- Only one observation detected

### 89
- Rows 17-20, 23-27, 30-32: Wrong day (Monday instead of Wednesday)

### 93
- 36 events total, 33 recorded
- Missed events: Wimbledon and 2 Leeds events on July 12
- Rows 25-26 wrong date

## Votes for Women

### 26
Counted 84 events, Gemini found 88

### 100
Counted 213 events, Gemini found 183  
Some cities are missing

### 223
Events are scans 8 and 9 and seems to be picked up (last meetings of scans = found by Gemini)
