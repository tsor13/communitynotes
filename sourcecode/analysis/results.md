## Plot and Categories

![alt text](image.png)
- Downloaded data as of Thursday, April 3rd
- Used cutoffs from the paper for the categories
  - Intercept >= 0.4: helpful
  - Intercept <= -0.08: not helpful
  - Factor <= -0.5: polarized left
  - Factor >= 0.5: polarized right

## Observations

### Helpful
Topics
- Many notes correct misinformation in a factual way with sources to back up their claims
- Topics: celebrities, current events, scientific facts, etc.
- A lot of the notes warn users of scams like phishing and giveaway scams
  - Incentive for a lot of people to mark this as helpful because it's not necessarily political or controversial
- Many also clarify or add context to notes that might be misleading
  - context behind images/videos/claims
  - background info
  - language/cultural context
- Some also contain public health info

Language
- Clear and concise
- Use a neutral, informative tone to communicate and use direct language to address the tweets (begin by stating the tweet is false, incorrect, misleading, etc)
- Provide links to reputable, established sources

### Unhelpful
Topics
- Also try to correct misinformation and provide more context/clarification
- Many address conspiracy theories (2020 election results, COVID)
- Main difference: these notes contain a lot of political/social commentary with strong biases/opinions on the matter at hand
- Many lacked sources

Language
- More aggressive
- Take a strong stance and many are opinions rather than corrections
- Contained notes with sarcasm, mocking, humor, anger, etc
- Likely marked as unhelpful because ...
  - Ineffective at communicating their point without bringing emotion into it
  - Irrelevant because they contained no useful additional info or context
  - Sometimes contributed to misinformation themselves

### Polarized Left/Right
Topics
- Correct misinformation
- Much greater emphasis on topics like politics, social issues, scientific claims, environment
- A lot use sources

Language
- More opinionated, like the unhelpful ones, but better at communicating
- Focus is more on exposing flaws or inaccuracies in the original tweet's arguments than on correcting them
- Less concise

### Center
- All the notes that didn't fall into the 4 categories above
- A lot use humor/satire to communicate point
  - Not as concise/clear to all
- Similar topics and sources as all the rest

## Sentiment Analysis

| Sentiment           | Helpful | Unhelpful | Polarized Left | Polarized Right | Center  |
|---------------------|:-------:|:---------:|:--------------:|:---------------:|:-------:|
| Neutral 😐          | 41.20%  | 38.40%    | 37.66%         | 32.63%          | 38.75%  |
| Negative 😞         | 17.94%  | 18.51%    | 18.98%         | 20.02%          | 18.35%  |
| Positive 🙂         | 16.80%  | 15.86%    | 14.57%         | 15.28%          | 16.55%  |
| Extremely Negative 😡 | 16.52% | 18.76%    | 21.49%         | 24.37%          | 18.22%  |
| Extremely Positive 😄 | 7.53%  | 8.47%     | 7.31%          | 7.70%           | 8.14%   |

OVERALL: Polarized notes were more negative, with the polarized right notes being the most negative, probably due to politics and controversial topics. Helpful and unhelpful were more neutral/negative leaning, with helpful notes containing the most neutral ones.

- Used VADER (found online that it's supposedly good for social media data because it does well with slang, emojis, etc.)
- Helpful
  - Highest proportion of neutral notes
  - Similar percentages of negative/positive/extremely negative
  - Majority of the sentiment isn't in the extremes
    - Even if notes are worded slightly positively/negatively, they are still marked as helpful if informative
- Unhelpful
  - Slightly less neutral and more negative/extremely negative but distributions still similar to helpful
  - Content probably has more of an impact than the tone
- Polarized Left
  - Highest percentage of extremely negative sentiment
  - Leans more negative than helpful/unhelpful and neutral: addressing more contentious topics so maybe due to disagreement with original tweet
- Polarized Right
  - Contains the most negative sentiment of all the groups: very critical/oppositional stance
  - Could also be due to the notes being more confrontational
- Center
  - middle ground-ish

## Sources
| Category         | Total Citations | Top Sources                                                                                                                                                                                                                   |
|------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **HELPFUL**      | 86,910          | 1. twitter.com (9,682) <br> 2. wikipedia.org (4,327) <br> 3. x.com (4,006) <br> 4. youtube.com (1,909) <br> 5. youtu.be (1,892) <br> 6. reuters.com (1,398) <br> 7. bbc.co.uk (1,006) <br> 8. vice.com (1,002) <br> 9. instagram.com (957) <br> 10. apnews.com (788) |
| **UNHELPFUL**    | 22,753          | 1. wikipedia.org (2,576) <br> 2. twitter.com (2,153) <br> 3. x.com (1,063) <br> 4. youtu.be (433) <br> 5. google.com (396) <br> 6. youtube.com (357) <br> 7. theguardian.com (228) <br> 8. cnn.com (198) <br> 9. reuters.com (190) <br> 10. nih.gov (185)          |
| **POLARIZED_LEFT** | 63,262         | 1. twitter.com (5,508) <br> 2. wikipedia.org (3,266) <br> 3. x.com (2,497) <br> 4. mhlw.go.jp (1,324) <br> 5. youtube.com (805) <br> 6. globo.com (676) <br> 7. togetter.com (606) <br> 8. youtu.be (591) <br> 9. google.com (574) <br> 10. cnn.com (557)          |
| **POLARIZED_RIGHT**| 76,914         | 1. wikipedia.org (5,182) <br> 2. twitter.com (4,473) <br> 3. x.com (2,139) <br> 4. reuters.com (1,304) <br> 5. nih.gov (1,168) <br> 6. apnews.com (1,152) <br> 7. theguardian.com (1,102) <br> 8. nytimes.com (809) <br> 9. cdc.gov (763) <br> 10. cnn.com (751)   |
| **CENTER**        | 347,644        | 1. twitter.com (30,273) <br> 2. wikipedia.org (21,955) <br> 3. x.com (12,090) <br> 4. mhlw.go.jp (8,470) <br> 5. reuters.com (4,618) <br> 6. youtu.be (4,222) <br> 7. youtube.com (4,184) <br> 8. togetter.com (4,034) <br> 9. nih.gov (3,390) <br> 10. apnews.com (3,287) |

General Observations
- Twitter, X, Wikipedia, Youtube are among the top sources across all categories
  - a lot of the notes reference other tweets made by relevant people or earlier/later tweets by the same author
  - convenient to cite info from wikipedia or link relevant videos

Helpful
- Cite more well-established news outlets as well as Instagram
  - Reuters, BBC, AP News
- Contains the most citations of the 4 main areas (other than center)
- 92% of these notes contained at least one citation

Unhelpful
- Least citations out of all of the notes
- 51% of notes contained citations
- Also contained google as one of the top 5 sources - people might have been relying on search results
  - Could be marked as unhelpful due to being generic
- Greater variety of sources, but not necessarily reliable sources

Polarized Left
- Good amount contain sources (75%)
- Contains more regionally specific sources
  - MHLW, Togetter from Japan
  - Globo from Brazil
  - Could be that more people from specific regions are submitting these notes than others

Polarized Right
- Good amount contain sources (78%)
- More health-related sources, like CDC,gov, NIH.gov
- Similar traditional news sources like Reuters, AP News, along with more politically-left leaning news sources like NY Times and The Guardian
- Could be addressing a lot of COVID-related or political tweets, which might explain why they were marked as less helpful while still citing good sources

## Readability
- All of them had around the same readability