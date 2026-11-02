# Open issues

What the paper does not state or what still needs a decision.

- **Instruction and consent texts are not printed** [decision, `iss_instructions`, m_oj_prompt]: Only example question wordings are printed; instruction screens, response-field label and consent text are not. Suggested: Supply the original (German) instruction screens or approve a minimal framing.
- **Food images are not available** [decision, `iss_images`, m_items]: The images are on OSF (not consulted) and are not in the paper. Suggested: Supply the images or accept a text-only replay. (p. 10, p. 9)
- **Filler text for the control group is missing** [blocking, `iss_filler`, m_filler]: The text is described as unrelated to the study but not printed. Suggested: Supply the filler text in materials/materials.json (control_filler_text). (p. 11)
- **City–value pairing, countries and flags are not printed** [decision, `iss_city`, m_info_irrelevant]: Only the Basel example and the city list with approximate longitudes are given. Suggested: Confirm the pairing and supply country names or flags. (p. 11, p. 33)
- **Counterbalancing scheme for item sets is not described** [decision, `iss_counter`, n_design]: Only 'comparable frequency' is stated. Suggested: Provide the assignment scheme. (p. 12)
- **Group allocation procedure and randomized n are not reported** [check, `iss_alloc`, n_participants]: Random assignment is stated; allocation ratio and the numbers randomized per group are not. Suggested: Provide allocation and randomized n. (p. 11)
- **Order of values in the information phase is not stated** [check, `iss_order`, s1]: The step description gives timing (8 s per item) but not the order. Suggested: State the order. (p. 11)
- **Two printed OJ wordings differ** [check, `iss_wording`, m_oj_prompt]: Figure 1 text reads 'How many grams of sugar are in a kiwi that weighs 85 grams?' and the procedure example reads 'How many grams of sugar does a kiwi that weighs 85 grams contain?'; both are translations. Suggested: Confirm which wording was shown. (p. 11, p. 9)
- **MCMC settings, coding and outlier reference are not stated** [check, `iss_mcmc`, a_tl]: Chains, iterations, task coding and the exact threefold-IQR rule are not given. Suggested: Provide the analysis code (OSF). (p. 14)
- **Outlier criterion not fully specified** [check, `iss_outlier`, a_exclusions]: Seventh exclusion step lacks the reference for the interquartile range. Suggested: Confirm in the analysis code. (p. 12, p. 13)
- **Cell means of |Δ| are not reported numerically** [check, `iss_cells`, res_fig2]: They appear only in Figure 2. Suggested: Obtain means from the OSF data. (p. 15)
- **Wording of compliance and display questions is not printed** [check, `iss_compliance`, m_engagement]: Described but not worded. Suggested: Supply wording if needed. (p. 12)
- **Units and phrasing for liquid and counted items are not printed** [check, `iss_units`, m_items]: Table A1 gives size in g or ml without marking which; examples only cover solid items with 'weighs … grams'. Suggested: Confirm the original item phrasing and units. (p. 30, p. 9)
- **Allocation procedure for random assignment is not stated** [check, `iss_alloc_rule`, n_design]: The paper says participants were randomly assigned but not how. Suggested: State the allocation procedure. (p. 11)
