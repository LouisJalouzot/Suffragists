# Known Errors
## Common Causes
Checked LJ: 1, 3, 4, 11, 29, 60, 77, 104, 105, 114, 130, 159, 165, 115, 183, 200, 235

Checked VR: 13, 167, 27, 84, 50, 184, 108, 221, 177

Notes: 
- 184 picked up two extra meetings from other suffrage societies that were listed next to the "forthcoming events"(minor error IMO).
- 221 picked up additional meetings because the list was very short (1 meeting). The meetings picked up are mostly relevant (although some come from other societies), and come from free text descriptives of meetings from the "News of Societies and Federations". Still OK in my book.
- 177 has a typo (OCR error, OK): "Orieff" in csv should be "Crieff"

### 200
Also picks up the table of "Coming events" which might also be relevant.

## VFW
Checked: 100 (meetings are scattered in small tables and seems like it is correctly picked up)

## Suffragettes
Checked: 26, 32, 36, 39, 44, 71, 83, 87, 89, 93

### 32
- Multiple event observations in single line causes speaker attribution errors
- Speakers wrong in rows 3-7, 9
- Row 12 contains two events, affecting speakers in rows 13-16, 18-19, 25, 27-29

### 36
- Speakers missing from row 4 in 3
- Row 5 truncates last word (pattern: missing text in last column)
- Speakers swapped in rows 8 and 9
- Row 28 contains 2 events, affects row 29
- Row 36 successfully handles cross-line events
- Row 56 missing Portsmouth
- Rows 61-62 picked up from outside table

### 39
Page in the background with some readable text messes up the clustering. Had to manually crop.
Picks up some events scattered in the text.

### 44
- Last column truncation pattern continues:
  - Row 2 misses speaker title "Mr HH"
  - Row 3 misses speaker name
  - Row 9 similar issue
- Row 3 has two events
- Rows 18-20, 27-30 date wrong (+1 day)
- Rows 31-33 correct but out of order

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
Checked: 1, 7, 80, 85, 118, 120, 140, 223

### 1
- 70 actual events, 80 detected
- Rows 71 & 73 duplicate event with wrong date (should be 7th)
- Row 72 wrong date (should be 6th)
- Rows 74-80 from outside table

### 7
- Rows 1-7 missing day (should be April 1st)
- Rows 8, 12-17 wrong date
- Rows 18-36 correct date
- Rows 37-39 missing day (should be April 2nd)
- Rows 40-47 wrong dates

### 80
- Good performance
- Missed 6 events on September 14th

### 85
- Excellent performance
- Only rows 94-97 wrong date (should be October 8th)

### 118
- Events picked up from outside table (rows 94-95)
- Rows 5-6 June/July appear from outside
- No May 21 events should exist
- May 22 correct
- May 23 has extra events
- Rows 87, 92 wrong date

### 120
- ~175 events expected (25/day for 7 days)
- Only 125 detected
- Multiple events per line (rows 24-43)

### 140
- Row 16 from elsewhere
- Row 21 wrong
- Missing 5 Saturday events
- Sunday missing first two events
- Row 40 wrong date and contains next day events
- October 18 missing first two and last two events
- Rows 56-62 wrong dates, inverse

### 223
- Rows 9-31 (except 26) date off by one day
- Missed events on 15th, first 2 on 16th, all 5 on 17th May
- Row 42 from below table
- Rows 43-52 wrong events from elsewhere
