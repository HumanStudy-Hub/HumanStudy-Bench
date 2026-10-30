# Prediction survey: laypeople forecast replication outcomes of 27 studies

`s1`

Online survey, two between-participant conditions (description only vs description plus Bayes factor); each participant rated 27 study descriptions plus one attention-check item.

## Hypotheses

### H1: above-chance accuracy

`h1`

- **Statement**: Laypeople can predict replication success with above-chance accuracy (accuracy above 50%). (p. 2)

### H2: confidence tracks replication effect size

`h2`

- **Statement**: Laypeople's confidence in replicability is positively correlated with the effect size observed in the replication. (p. 2)

### H3: evidence information improves prediction

`h3`

- **Statement**: Including information on the strength of the original evidence (the Bayes factor) improves prediction performance. (p. 2, p. 3)

## Participants

### Lay participants

`p1`

- **Setting**: Online survey built in Qualtrics (p. 4)
- **Roles [derived]**: Forecasters (every participant rated all studies); no confederates (p. 4)
  - how it was derived: Derived from the procedure: all participants rate the same set of studies.
- **Recruitment**: MTurk (n = 83), first-year psychology students at the University of Amsterdam (n = 138), social media such as Facebook (n = 36); 257 in total (p. 3)
- **Eligibility**: Laypeople: no Ph.D. in psychology (graduate students count as laypeople); participants reporting a Ph.D. in psychology were redirected to the end of the survey (p. 4)
- **Recruited**: 257 (p. 3)
- **Analysed**: {"total": 233, "description-only": 123, "description-plus-evidence": 110} (p. 3)
- **Exclusions**: 3 familiar with the replication projects, 22 failed the attention check, none understood less than 50% of descriptions; 99 predictions (1.6%) dropped because the participant did not understand the description (p. 4)
- **Characteristics**: 72% (167) understood all descriptions; Dutch participants could read the descriptions in Dutch (p. 4)
- **Sampling unit [derived]**: Individual participant (p. 5)
  - how it was derived: Brier-score and accuracy analyses treat the individual participant as the unit; the correlation analysis uses the study.
- **Compensation**: MTurk: financial compensation; UvA students: research credits; social media: raffle for a voucher from a Dutch web shop (p. 3)
- **Sampling plan**: Minimum of 103 observations per condition for an 80% probability of a Bayes factor > 10 (delta = 0.5, Cauchy r = .707, two independent groups); data collection continued to April 22, 2019 (p. 3)

## Design

### Design

`d1`

- **Type**: Between-participant, two conditions; each participant rates the same 27 studies (plus one attention-check item) (p. 4)
- **Assignment**: Random assignment to description-only or description-plus-evidence (p. 4)
- **Order**: Order of the studies randomized across participants (not randomized for 12 Dutch social-media participants because of a programming error) (p. 4, p. 17)
- **Manipulation check**: Attention check: a bogus item instructing 'no' and a confidence of 75% (p. 4)

## Conditions

### Description only

`c-desc`

- **What participants see**: Instructions on replication and, for each study, the plain-language description only (p. 3, p. 4)
- **Analysed n**: 123 (p. 3)

### Description plus evidence

`c-evid`

- **What participants see**: The same instructions plus an explanation of the Bayes factor and its verbal categories; each description is followed by the study's Bayes factor and a verbal label (p. 4)
- **Analysed n**: 110 (p. 3)

## Materials

### Replication instructions

`m-instr`

- **Instruction text (verbatim)** (verbatim, p. 4):
  > You will be asked whether you think that the described study will replicate. This means: if an independent lab will do this study again with a large number of participants, using the same materials, will they find convincing evidence for the same effect? If the effect really exists, it should be found by a different lab. However, it seems that not all studies can be replicated, because some results are based on coincidence, or poorly designed or executed studies.
- **Wording around the quoted instruction**: Not stated in the paper.

### Bayes factor explanation

`m-bfexp`

Conditions: Description plus evidence

- **Explanation (verbatim)** (verbatim, p. 4, p. 17):
  > A Bayes factor (BF) is the degree to which evidence is found for the existence of the effect, based on the data at hand. For instance, if BF = 2, the data suggest that it is 2 times more likely that the effect is present, than that there is no effect.
- **Verbal categories shown with the explanation**: Not stated in the paper.

### The 27 study descriptions

`m-items`

- **Layout of each page [to check]**: Headings 'Research question', 'How was this measured?', 'What was found?', and (evidence condition only) 'Evidence strength in Bayes factor' with 'BF = x' and 'This qualifies as <weak|moderate|strong|extreme> evidence.'; counter 'k / 28' (p. 19)
- **Descriptions (verbatim, English)** (verbatim, p. 1):
  > [{"key": "aviezer", "study": "Aviezer, Trope, and Todorov (2012)", "description": "Can people distinguish between positive and negative emotions based on the body alone? Participants looked at images of athletes that either just scored or just lost a point in a competition. Without seeing any facial expressions, they then judged whether the image depicted a positive or negative emotion. Participants correctly recognized the underlying emotions based on the image of the body alone. That is, they rated the images of winning athletes as more positive then the images of losing athletes."}, {"key": "balafoutas", "study": "Balafoutas and Sutter (2012)", "description": "Does preferential treatment encourage women to enter competition with others? In a game, male and female participants chose whether they wanted to compete against each other, or whether they wanted to get judged based on their individual performance. In one group, women received a preferential treatment if they entered the competition, that is, one point was automatically added to their score. In the other group, women did not receive a preferential treatment. If women received a preferential treatment, then they would choose to enter a competition with others more often."}, {"key": "derex", "study": "Derex, Beugin, Godelle, and Raymond (2013)", "description": "Are bigger groups characterized by more cultural diversity? A computer game was set up to be a multiplayer game, consisting of either 2, 4, 8, or 16 players. In this game the players had to collect points by either building arrowheads (a simple task), or building fishing nets (a difficult task). The more players were playing the game, the more likely it was that they increased the cultural diversity within the game (i.e., by building both items)."}, {"key": "duncan", "study": "Duncan, Sadanand, and Davachi (2012)", "description": "Does the detection of new objects improve people’s ability to detect subtle changes in similar objects? Participants looked at a series of object images. Afterwards, they had to look again at a series of objects and judged whether they were ‘old’ (seen before), ‘new’, or ‘similar’ (looked like something seen before, but not the same). If ‘new’ objects were shown before ‘similar’ objects, then more participants recognized the ‘similar’ objects correctly."}, {"key": "gervais", "study": "Gervais and Norenzayan (2012)", "description": "Does analytic thinking trigger religious disbelief? Participants looked at an image. One group looked at an image of “The Thinker”, that is a sculpture of a person in a thinking position. The other group looked at an image which depicted a sculpture of a discus thrower. Afterwards, they answered questions about their religiosity. Participants who had seen “The Thinker” reported lower religious belief compared to people who had seen the sculpture of a discus thrower."}, {"key": "gneezy", "study": "Gneezy, Keenan, and Gneezy (2014)", "description": "Do people avoid charities that dedicate a high percentage of donations to administrative and fundraising costs? People chose between donating to either a drinking water charity, or a volunteering charity. One group was told that 50% of the donations to the water charity were used to cover administrative and fundraising costs. The other group was told that all costs for the water charity were already covered. If administrative and fundraising costs were covered, people were more likely to donate to the water charity than to the volunteering charity, compared to when the costs were not covered."}, {"key": "hauser", "study": "Hauser, Rand, Peysakhovich, and Nowak (2014)", "description": "Do people preserve common resources for future generations if they make collective decisions? In a game, participants chose how many resources they would extract from a pool, and how many they wanted to preserve for future game players. In one group, each participant made their own decision on how many resources they want to extract from the pool. In the other group, participants voted for the combined amount of resources they would extract. If people had to vote on the combined amount of resources they would extract from the pool, more resources were preserved for future generations of game players."}, {"key": "bauer", "study": "Bauer, Wilkie, Kim, and Bodenhausen (2012)", "description": "Do people with a “materialism mindset” trust others less? Participants read a dilemma about distributing resources related to water conservation. In one group, other parties involved in the water conservation problem were referred to as “consumers”. In the other group, these parties were referred to as “individuals”. Afterwards, participants had to report their trust towards the other parties involved in the water conservation problem. If other parties involved in the water conservation problem were referred to as ‘consumers’, participants trusted these parties less."}, {"key": "critcher", "study": "Critcher and Gilovich (2008)", "description": "Are people’s estimations of numbers influenced by unrelated numbers that are incidentally present in the environment? Participants read a description about a new cell phone. In one group, the cell phone was called P17. In the other group, the cell phone was called P97. Afterwards, participants predicted its proportion of sales. If the cell phone was called P97, participants predicted a higher proportion of sales."}, {"key": "karpicke", "study": "Karpicke and Blunt (2011)", "description": "Is the learning strategy ‘retrieval practice’ more efficient than the learning strategy ‘concept mapping’? Participants studied a science text using a certain learning strategy. In one group, participants used the learning strategy ‘retrieval practice’, which consists of learning - testing - learning. In the other group, participants used the learning strategy ‘concept-mapping’, in which participants create a diagram with nodes and links to connect different concepts. One week later participants were tested on this text. If participants used the learning strategy ‘retrieval practice’, they performed better on a memory test."}, {"key": "anderson", "study": "Anderson, Kraus, Galinsky, and Keltner (2012)", "description": "Do social comparisons influence people’s well-being? Participants read a description of a person and were instructed to think about similarities and differences between them. In one group, the person was described as respected and liked by all their social groups. In the other group, the person was described as not respected and liked in any of their social groups. Afterwards, participants answered questions about their well-being. If participants compared themselves with a highly respected and liked person, participants reported lower well-being."}, {"key": "giessner", "study": "Giessner and Schubert (2007)", "description": "Is perceived power related to the vertical location of a person? Participants studied a schematic display of the hierarchy within an organization, including a manager and his team. In one group, the vertical line connecting the manager to the team was long (i.e., 7 cm). In the other group, the vertical line was short (i.e., 2 cm). Afterwards, participants estimated how much power they thought the manager held within the organization. If the vertical line connecting the manager to the team was long, participants estimated that the manager held more power within the organization."}, {"key": "risen", "study": "Risen and Gilovich (2008)", "description": "Do people belief that tempting fate leads to negative consequences? Participants imagined a scenario in which they would come to a lecture in which the professor picks out one student to answer a difficult question in front of the entire class. In one group, participants imagined that they tempted fate by coming to the lecture unprepared. In the other group, participants imagined that they came to the lecture prepared. Afterwards, participants had to estimate how likely it was that they would get chosen. If participants imagined that they tempted fate, they thought it was more likely that they would get chosen by the professor to answer a difficult question in front of the entire class."}, {"key": "shafir", "study": "Shafir (1993)", "description": "Do people find positive characteristics more important in decisions in which they are awarding something, and find negative characteristics more important in decisions in which they are denying something? Participants imagined that they were members of a jury, who had to decide which parent would obtain custody for a child. One parent was described as having average features concerning, for instance, income, work-related absence, and relationship to the child. The other parent had both extreme positive characteristics (e.g., very close relationship to the child), but also extreme negative characteristics (e.g., much work-related absence). In one group, participants were asked to which parent they would award custody for the child. In the other group, participants were asked to which parent they would deny custody for the child. In both groups, participants selected the extreme parent more often than the average parent."}, {"key": "zaval", "study": "Zaval, Keenan, Johnson, and Weber (2014)", "description": "Do people’s concerns about global warming change when the concept of heat or cold is active in their minds? Participants performed a task in which they had to unscramble sentences. In one group, participants read sentences that contained words related to hot temperatures. In the other group, people read sentences that contained words related to cold temperatures. Afterwards, participants reported the degree in which they were concerned about global warming. If participants read sentences that contained words related to hot temperatures, participants were more concerned about global warming."}, {"key": "tversky", "study": "Tversky and Gati (1978)", "description": "Are people’s judgements of how similar two concepts are influenced by the order in which they were mentioned? Participants rated how similar two countries were to each other (e.g., “How similar is the USA to Lebanon?”). One of the countries was well-known to the participants (e.g., the USA). The other country was less familiar to the participants (e.g., Lebanon). In one group, the well-known country was mentioned first. In the other group, the less known country was mentioned first. If the well-known country was mentioned first, participants judged the two countries as less similar."}, {"key": "kidd", "study": "Kidd and Castano (2013)", "description": "Can reading literary fiction improve people’s understanding of other people’s emotions? Participants read a short text passage. In one group, the text passage was literary fiction. In the other group, the text passage was non-fiction. Afterwards, participants had to identify people’s expressed emotion (e.g., happy, angry) based on images of the eyes only. Participants were better at recognizing the correct emotion after reading literary fiction."}, {"key": "shah", "study": "Shah, Mullainathan, and Shafir (2012)", "description": "Does poverty drain people’s attention? Participants played the game “Wheel of Fortune”, a game in which people have to guess letters in word puzzles. In one group, participants were given 6 chances per round to guess letters (i.e., ‘poor’ players). In the other group, participants were given 20 chances per round to guess letters (i.e., ‘rich’ players). Afterwards, they completed an attention task. If participants were given few chances per round to guess letters, they performed worse in the subsequent attention task."}, {"key": "zhong", "study": "Zhong and Liljenquist (2006)", "description": "Does feeling morally dirty increase people’s need to wash themselves? Participants hand copied a story written in the first person. In one group, participants rewrote an unethical short story about sabotaging a co-worker. In the other group, participants rewrote an ethical short story about helping a co-worker. Afterwards, participants expressed their desire for cleaning products (e.g., soap, toothpaste). If participants rewrote an unethical story, they had a higher desire for cleansing products."}, {"key": "kovacs", "study": "Kovacs, Téglás, and Endress (2010)", "description": "Do beliefs of others influence people’s actions, even if these beliefs are irrelevant? Participants watched a short cartoon. In the cartoon, a character places a ball behind a box, which then rolls away. Then, the character lifts the box and reveals the ball, which returned unexpectedly. Participants were instructed to press a button as soon as they detected the ball. In one group, the character saw the ball rolling away. In the other group, the character did not see the ball rolling away. If the character believed that the ball was behind the box, participants detected the ball faster."}, {"key": "lee", "study": "Lee and Schwarz (2010)", "description": "Does washing hands weaken people’s urge to justify their choice for a non-preferred item? Participants ranked 10 CDs based on how much they would like to own them. Then, the participants could choose to keep their 5th or 6th choice. After the choice, participants were asked to evaluate a soap and then to rank the CDs again. In one group, participants evaluated the soap after they washed their hands with it. In the other group, participants evaluated the soap without washing their hands with it. If participants evaluated the soap without washing their hands with it, they increased their preference for their chosen CD."}, {"key": "morewedge", "study": "Morewedge, Huh, and Vosgerau (2010)", "description": "Do people want to eat less food, after they repeatedly imagined eating it? Participants had to imagine 33 repetitive actions, one at a time. In one group, participants imagined eating 30 M&Ms and then inserting 3 coins in a laundry machine. In the other group, participants imagined inserting 33 coins in a laundry machine. Afterwards, participants could eat from a bowl containing M&Ms. If participants imagined eating 30 M&Ms, they ate fewer M&Ms from the bowl."}, {"key": "nishi", "study": "Nishi, Shirado, Rand, and Christakis (2015)", "description": "Are groups less likely to reduce inequality if the group members know the wealth of each other? In a game, participants could share resources with co-players or keep it to themselves. At the start of the game, the resources were distributed unevenly between the players. In one group, participants saw the amount of resources of other players. In the other group, participants only saw their own resources. If participants knew the amount of resources from other players, the inequality between the players remained higher."}, {"key": "pyc", "study": "Pyc and Rawson (2010)", "description": "When learning new words, does taking a test help people to remember links between the words and their meaning? Participants learned 48 Swahili words by writing down links between each Swahili word and its meaning. In one group, the learning strategy of the participants involved learning - testing - learning. In the other group, the learning strategy of the participants involved learning - relearning, without testing. One week later, participants did a memory test in which they got the instruction to write down the meaning of the Swahili word, and their self-generated link between the word and its meaning. If participants used a learning strategy that involved learning - testing - learning, they remembered more of their self-generated links between the words and their meaning."}, {"key": "sparrow", "study": "Sparrow, Liu, and Wegner (2011)", "description": "Do difficult questions activate the concepts of \"Google” and computers in people’s minds? Participants answered knowledge questions. In one group, participants answered difficult questions. In the other group, participants answered easy questions. Afterwards, they performed a reaction time task where they had to ignore computer-related words that were presented on the screen. If participants answered difficult questions, their reaction times were slower when computer-related words were present on the screen (which indicates that the concept of computers was active in their minds)."}, {"key": "wilson", "study": "Wilson et al. (2014)", "description": "Do people enjoy doing nothing? Participants spent a short amount of time by themselves in an empty room. In one group, participants were instructed to spend their time on a non-social activity (e.g., listening to music, reading a book, surfing the Web). In the other group, participants were instructed to entertain themselves with their thoughts (without any external activity). Afterwards participants answered questions about how enjoyable this experience was. If participants had just their thoughts to entertain themselves, they enjoyed themselves less."}, {"key": "alter", "study": "Alter, Oppenheimer, Epley, and Eyre (2007)", "description": "Do people’s logic skills improve if the concept of analytic thinking is active in their minds? Participants solved difficult logic problems. In one group, the logic problems were printed in a font that was hard to read (activating the analytical mindset). In the other group, the logic problems were printed in a font that was easy to read. If the logic problems were printed in a font that was hard to read, participants solved more of the logic problems correctly."}]
- **Bayes factor and verbal label shown in the evidence condition (verbatim)** (verbatim, p. 1, p. 9):
  > [{"key": "aviezer", "bf": "larger than 1 million", "verbal": "extreme"}, {"key": "balafoutas", "bf": "4.6", "verbal": "moderate"}, {"key": "derex", "bf": "21111", "verbal": "extreme"}, {"key": "duncan", "bf": "11.4", "verbal": "strong"}, {"key": "gervais", "bf": "2.1", "verbal": "weak"}, {"key": "gneezy", "bf": "11.9", "verbal": "strong"}, {"key": "hauser", "bf": "larger than 1 million", "verbal": "extreme"}, {"key": "bauer", "bf": "105", "verbal": "extreme"}, {"key": "critcher", "bf": "1.2", "verbal": "weak"}, {"key": "karpicke", "bf": "484", "verbal": "extreme"}, {"key": "anderson", "bf": "11.8", "verbal": "strong"}, {"key": "giessner", "bf": "1.9", "verbal": "weak"}, {"key": "risen", "bf": "1.5", "verbal": "weak"}, {"key": "shafir", "bf": "1.9", "verbal": "weak"}, {"key": "zaval", "bf": "1.3", "verbal": "weak"}, {"key": "tversky", "bf": "6.4", "verbal": "moderate"}, {"key": "kidd", "bf": "3.5", "verbal": "moderate"}, {"key": "shah", "bf": "1.5", "verbal": "weak"}, {"key": "zhong", "bf": "4", "verbal": "moderate"}, {"key": "kovacs", "bf": "2.4", "verbal": "weak"}, {"key": "lee", "bf": "4", "verbal": "moderate"}, {"key": "morewedge", "bf": "5.4", "verbal": "moderate"}, {"key": "nishi", "bf": "7.1", "verbal": "moderate"}, {"key": "pyc", "bf": "2.6", "verbal": "weak"}, {"key": "sparrow", "bf": "15.5", "verbal": "strong"}, {"key": "wilson", "bf": "422", "verbal": "extreme"}, {"key": "alter", "bf": "1.5", "verbal": "weak"}]
- **Dutch version [to check]**: Table 2 of the supplement holds Dutch descriptions; the replay uses the English table only (p. 4, p. 1)

### Attention-check (bogus) item

`m-attn`

- **Text of the bogus study description**: Not stated in the paper.
- **Pass rule**: Pass if 'no' is selected and confidence is between 70% and 80% (p. 4)

### Response items

`m-qs`

- **Understanding tick box** (verbatim, to check, p. 4, p. 19):
  > I don't understand the study description
- **Replication question** (verbatim, to check, p. 4, p. 19):
  > Will this study replicate? (Yes / No)
- **Confidence question** (verbatim, to check, p. 4, p. 19):
  > How confident are you about your decision? (slider labelled 0%, 50%, 100%; handle starts at 50%)
- **Wording of the closing familiarity question**: Not stated in the paper.
- **Scale meaning of 0-100 slider for a 'no' prediction**: Not stated in the paper.

## Procedure

Order of steps:

- Random assignment to a condition (p. 4)
- Read the explanation of replication (participant sees: Replication instructions) (p. 4)
  - when description-plus-evidence condition:
    - Read the Bayes factor explanation (participant sees: Bayes factor explanation) (p. 4)
      - conditions: Description plus evidence
    - Repeated: Rate 27 studies plus the attention-check item, one per page, in random order (participant sees: The 27 study descriptions) (p. 4)
      - Read the study description (and, in the evidence condition, its Bayes factor and verbal label) (participant sees: The 27 study descriptions) (p. 3, p. 4)
      - Optionally tick that the description was not understood (p. 4)
      - Predict whether the study will replicate (yes/no) (p. 4)
      - Rate confidence in the decision (0-100) (p. 4)
      - (back to: Optionally tick that the description was not understood)
      - (back to: Predict whether the study will replicate (yes/no))
      - (back to: Rate confidence in the decision (0-100))
    - Answer whether already familiar with ML2 or SSRP (p. 4)
  - when description-only condition:
    - (back to: Rate 27 studies plus the attention-check item, one per page, in random order)

### Survey procedure

`pr1`

- **Task**: For each described study, judge whether a large independent replication would find convincing evidence for the same effect (yes/no) and report confidence (0-100) (p. 4)
- **Actions**: Read instructions (plus Bayes factor explanation in the evidence condition); view 27 + 1 studies, one per page, in random order; tick 'did not understand' if applicable; answer yes/no; set confidence; at the end answer familiarity question (p. 4)
- **What is visible**: Only the current page's description (and, in the evidence condition, its Bayes factor with verbal label); no replication outcomes, author names, titles or quotes (p. 4)
- **Feedback [to check]**: None reported (p. 4)
- **Payoff [to check]**: No accuracy-contingent payoff reported (p. 3)

## Variables

### Replication prediction

`v-pred`

- **Role [derived]**: dv (p. 4)
  - how it was derived: Prediction is scored against the replication outcome.
- **Type**: binary (yes/no) (p. 4)
- **Observation unit [derived]**: participant x study (p. 4, p. 5)
  - how it was derived: One yes/no per participant per study page.

### Confidence

`v-conf`

- **Role [derived]**: dv (p. 4)
  - how it was derived: Collected for each prediction.
- **Type**: continuous slider 0-100 (p. 4)
- **Observation unit [derived]**: participant x study (p. 4)
  - how it was derived: One slider value per page.
- **Signed confidence**: Figures plot signed confidence from -100 (certain failure) to 100 (certain success) (p. 6)

### Condition

`v-cond`

- **Role**: iv (p. 7)
- **Type**: binary: description only vs description plus evidence (p. 4)
- **Observation unit [derived]**: participant (p. 4)
  - how it was derived: Condition is assigned once per participant.

### Replication outcome of each study

`v-outcome`

- **Role [derived]**: criterion (p. 5)
  - how it was derived: Prediction accuracy is computed against it.
- **Definition**: Success or failure following SSRP and ML2 primary criteria: a significant effect in the same direction as the original study (p. 15)
- **Values by study [derived, to check]**: {"aviezer": "replicated", "balafoutas": "replicated", "derex": "replicated", "duncan": "replicated", "gervais": "not replicated", "gneezy": "replicated", "hauser": "replicated", "bauer": "replicated", "critcher": "not replicated", "karpicke": "replicated", "anderson": "not replicated", "giessner": "not replicated", "risen": "replicated", "shafir": "not replicated", "zaval": "not replicated", "tversky": "not replicated", "kidd": "not replicated", "shah": "not replicated", "zhong": "not replicated", "kovacs": "replicated", "lee": "not replicated", "morewedge": "replicated", "nishi": "replicated", "pyc": "replicated", "sparrow": "not replicated", "wilson": "replicated", "alter": "not replicated"} (p. 6, p. 5, p. 3)
  - how it was derived: Read from the light/dark shading of Fig. 1 (page 6); consistent with the reported 14 replicated / 13 not (top 10: 9 successes, Anderson et al. fails; middle 13: 5 successes; bottom 4: none).
- **Observation unit [derived]**: study (p. 5)
  - how it was derived: One outcome per study.

### Replication effect size (r)

`v-reff`

- **Role**: criterion for the confidence correlation (p. 10)
- **Values by study**: Not stated in the paper.

### Original-study Bayes factor

`v-bf`

- **Role [derived]**: iv (stimulus information) (p. 4)
  - how it was derived: Shown only in the description-plus-evidence condition.
- **Values**: See the 'bayes-factors' field of the study descriptions material (p. 1)

### Individual log Brier score

`v-brier`

- **Role**: dv (condition comparison) (p. 6, p. 7)
- **Transformation**: Brier score combining accuracy and indicated certainty, log-transformed (p. 7)
- **Observation unit**: participant (p. 5)
- **Exact mapping from yes/no + slider to a probability**: Not stated in the paper.

## Analysis

### Quality check

`a-quality`

- **Aim**: Ensure data are not worse than chance before confirmatory analyses (p. 5)
- **Method**: Bayesian hierarchical model; Hr1: omega < .5 against encompassing He (omega ~ Beta(1,1)); encompassing-prior Bayes factor (p. 5)
- **Criterion**: Data inappropriate if strong evidence that overall accuracy is below 50% (p. 5)

### Condition difference in Brier scores

`a-brier`

- **Aim**: Test whether adding the Bayes factor lowers Brier scores (p. 6)
- **Method**: Bayesian independent-samples t test on log Brier score with condition as grouping variable; one-sided H-: delta < 0 vs H0: delta = 0 (p. 7)
- **Prior specification**: Not stated in the paper.

### Group-level accuracy per condition

`a-acc`

- **Aim**: Test whether accuracy exceeds chance (50%) in each condition (p. 8)
- **Method**: Bayesian hierarchical model of accuracy; restricted hypotheses Hr2, Hr3: omega > .5 vs H0: omega = .5, omega = mode of the group-level distribution; confidence is ignored (p. 8)
- **Unit**: Group-level accuracy of the average individual (hierarchical shrinkage), not aggregate accuracy of the group (p. 8)

### Confidence vs replication effect size

`a-corr`

- **Aim**: Test whether study-level confidence relates to replication effect size (p. 10)
- **Method**: Bayesian Spearman correlation per condition between study-level mean confidence and replication effect size; H0: rho = 0 vs one-sided rho > 0 (p. 10, p. 5)

### Signal detection analysis (exploratory)

`a-sdt`

- **Aim**: Separate discriminability from response bias (p. 10)
- **Method**: Bayesian hierarchical equal-variance Gaussian SDT model per condition; AUC derived from discriminability; no statistical tests (p. 11, p. 12)

## Reported results

### Quality check result

`res-quality`

- **Kind**: reported (p. 5)
- **Test**: Evidence for the encompassing hypothesis approached infinity; data passed the quality check (p. 5)

### Brier score difference

`res-brier`

- **Kind**: reported (p. 7)
- **Test**: Overwhelming evidence that description-plus-evidence outperformed description-only (BF-0 = 1.0 x 10^10) (p. 7)
- **Effect**: Median delta = -0.96, 95% CI [-1.23, -0.68] (p. 7)

### Group accuracy

`res-acc`

- **Kind**: reported (p. 9)
- **Descriptives**: {"description-only": "omega .59, 95% CI [.57, .60]", "description-plus-evidence": "omega .67, 95% CI [.65, .69]"} (p. 9)
- **Test**: Extreme support for accuracy above chance in both conditions (BFr20 = 4.4 x 10^7; BFr30 = 5.6 x 10^22) (p. 9)

### Confidence-effect size correlation

`res-corr`

- **Kind**: reported (p. 10)
- **Effect**: {"description-only": "median rho .61, 95% CI [.34, .77]", "description-plus-evidence": "median rho .77, 95% CI [.57, .87]"} (p. 10)

### SDT results

`res-sdt`

- **Kind**: reported (p. 12)
- **AUC**: {"description-only": "0.62, 95% CI [0.60, 0.65]", "description-plus-evidence": "0.74, 95% CI [0.72, 0.77]"} (p. 12)
- **Bias**: Negative bias in both conditions: tendency to overestimate replicability (p. 11)

## Simulation scope

### Simulation scope

`sim1`

- **Steps [implementation]**: Possible: instructions, 27 study pages in random order, yes/no and confidence per study, per-condition scoring. Not run: Dutch version, human recruitment channels, attention-check item (text unavailable), familiarity question (wording unavailable). Pages are presented in one prompt per AI participant, in the randomized order, with page counters.
- **Roles [implementation]**: Every original participant is an AI participant; no scripted roles
- **Conditions [implementation]**: Both conditions can be run exactly; the evidence condition adds the Bayes factor explanation and per-study BF lines
- **Outcomes [implementation]**: Accuracy, log Brier score (formula an adaptation), signal-detection summary; confidence-effect size correlation cannot be computed without the replication effect sizes
- **Limits [implementation]**: AI participants are not the laypeople sampled (MTurk, students, social media); models may know these famous studies' replication outcomes; Bayesian models are replaced by frequentist summaries offline; agreement with the paper does not show people would behave the same way

## Open issues

- **Attention-check item text not printed** [decision, `i-attn`, m-attn]: Only the instruction (answer 'no', confidence 75%) is described. Suggested: Supply the printed item from the OSF Qualtrics file (.qsf) or confirm the replay should omit it. (p. 4)
- **Instruction text beyond the quoted paragraph** [check, `i-lead`, m-instr]: Only the quoted replication explanation is printed. Suggested: Supply any further participant-visible text from the OSF materials. (p. 4)
- **Verbal categories of Bayes factors not listed** [check, `i-cats`, m-bfexp]: Paper cites the categories (Lee & Wagenmakers, 2013) but does not print them. Suggested: Supply the category table from the survey file. (p. 4)
- **Familiarity question wording unavailable** [check, `i-fam`, m-qs]: Wording and options not printed. Suggested: Supply wording from the survey file. (p. 4)
- **Meaning of the slider after a 'no' answer** [check, `i-slider`, m-qs]: The paper says confidence in the decision; the attention check implies confidence in the chosen answer. Suggested: Confirm from the survey file or the preregistration. (p. 4)
- **Brier score formula not stated** [decision, `i-brier`, v-brier]: The conversion of yes/no plus confidence to a probability and the aggregation over studies are not specified. Suggested: Confirm the mapping from the preregistration or analysis code on OSF. (p. 6)
- **Model priors and Bayesian model details not in the paper** [check, `i-priors`, a-brier]: They are in the Supplemental Material and OSF code. Suggested: Supply the Supplemental Material to reproduce the Bayesian models. (p. 7)
- **Replication effect sizes not printed** [check, `i-reff`, v-reff]: They come from Camerer et al. (2018) and Klein et al. (2018) and appear only in Fig. 7. Suggested: Supply the effect sizes from the OSF data or the replication reports. (p. 10)
- **Replication outcomes read from figure shading** [check, `i-outcomes`, v-outcome]: Per-study outcomes are only shown by shading in Fig. 1. Suggested: Confirm against Camerer et al. (2018) and Klein et al. (2018). (p. 6, p. 3)
- **Item layout taken from a screenshot** [check, `i-layout`, m-items]: The heading structure and the response items are read from images in the supplement, and the description text is a single paragraph there. Suggested: Confirm the split of each description against the survey file. (p. 19)
- **Dutch version not replayed** [check, `i-dutch`, m-items]: Dutch participants could choose a Dutch version (DeepL-assisted translation). Suggested: None needed unless a Dutch replay is wanted. (p. 4)
- **Feedback and payoff not reported** [check, `i-feedback`, pr1]: The paper does not say whether participants saw feedback or had accuracy-based payoffs. Suggested: Confirm from the survey file. (p. 4)
- **Payoff contingency not reported** [check, `i-pay`, pr1]: Only flat compensation is reported. Suggested: Confirm from the survey file. (p. 3)
