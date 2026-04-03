clear
import excel "C:\Users\sstoffel\Documents\temp_lit\decoy\DescriptiveAnalysis.xlsx", sheet("Sheet1") firstrow
rename ParticipateYes1NO0 part
recode part 1=0 if ParticipantNumber=="14"
recode part 1=0 if ParticipantNumber=="16"
recode part 1=0 if ParticipantNumber=="123"
recode part 1=0 if ParticipantNumber=="127"
la def part 0 "No" 1 "Yes"
la val part part
la var part "Completed the survey"
gen cond1=0 if D==1
replace cond1=1 if D==2 | D==3
la def cond1 0 "Control" 1 "Decoy"
la val cond1 cond1
la var cond1 "Experimental condition"
rename D cond2
recode cond2 1=0 2=1 3=2
la def cond2 0 "Control" 1 "Decoy: target 1st" 2 "Decoy; decoy 1st"
la val cond2 cond2
la var cond2 "Experimental condition"
gen agecat=0 if AgeRange=="20-24"
replace agecat=1 if AgeRange=="25-29"
replace agecat=2 if AgeRange=="30-35"
la def agecat 0 "20-24" 1 "25-29" 3 "30-35"
la val agecat agecat
la var agecat "Age category"
gen gender=0 if M=="Male"
replace gender=1 if M=="Female"
la def gender 0 "Male" 1 "Female"
la val gender gender
la var gender "Gender"
gen edu=0 if EducationLevel=="2"
replace edu=1 if EducationLevel=="1"
la def edu 0 "Bachelor's degree" 1 "Graduate or professional degree"
la val edu edu
la var edu "Education level"
gen ethn=0 if Race=="5"
replace ethn=1 if Race=="1"
replace ethn=2 if Race=="6"
replace ethn=3 if Race=="2" | Race=="3" | Race=="4"
la def ethn 0 "White" 1 "Asian" 2 "Black" 3 "Other"
la val ethn ethn
la var ethn "Ethnicity"
gen rel=0 if Religion=="3"
replace rel=1 if Religion=="1" | Religion=="4" | Religion=="5" | Religion=="6"
replace rel=2 if Religion=="2"
la def rel 0 "Christian" 1 "Ohter religion" 2 "No religion"
la val rel rel
la var rel "Religion"
save "C:\Users\sstoffel\Documents\temp_lit\decoy\exp.dta", replace 
////////////////////////////////////////////////////////////////////////////////
clear
use "C:\Users\sstoffel\Documents\temp_lit\decoy\exp.dta"
** Table 1 Description of the study sample of the experiment (N=203)
tab agecat cond1 if ValidEmailYes1No0==1, chi2  col
tab gender cond1 if ValidEmailYes1No0==1, chi2 col
tab ethn cond1 if ValidEmailYes1No0==1, chi2 exact col
tab rel cond1 if ValidEmailYes1No0==1, chi2 exact col
tab edu cond1 if ValidEmailYes1No0==1, chi2 col
** Table 2. Participation rates across the two experimental conditions (N=203)
tab part cond1 if ValidEmailYes1No0==1, chi2 col
** Table 3. Participation rates within the decoy condition (N=102)
tab part cond2 if ValidEmailYes1No0==1, chi2 col
** Table 4. Binary logistic regression on completing the survey
quietly logit part i.cond1 if ValidEmailYes1No0==1, or
outreg using reg1.doc, stats(e_b e_ci) nosubstat summstat(N) ctitle(Variable, Odds ratio, CI) replace 
quietly logit part i.cond1 i.agecat i.gender i.ethn i.rel i.edu if ValidEmailYes1No0==1, or
outreg using reg1.doc, stats(e_b e_ci) nosubstat merge summstat(N) ctitle(Variable, Odds ratio, CI) 
// 
quietly logit part i.agecat if ValidEmailYes1No0==1, or
outreg using reg2.doc, stats(e_b e_ci) nosubstat summstat(N) ctitle(Variable, Odds ratio, CI) replace 
quietly logit part i.gender if ValidEmailYes1No0==1, or
outreg using reg3.doc, stats(e_b e_ci) nosubstat summstat(N) ctitle(Variable, Odds ratio, CI) replace 
quietly logit part i.ethn if ValidEmailYes1No0==1, or
outreg using reg4.doc, stats(e_b e_ci) nosubstat summstat(N) ctitle(Variable, Odds ratio, CI) replace 
quietly logit part i.rel if ValidEmailYes1No0==1, or
outreg using reg5.doc, stats(e_b e_ci) nosubstat summstat(N) ctitle(Variable, Odds ratio, CI) replace
quietly logit part i.edu if ValidEmailYes1No0==1, or
outreg using reg6.doc, stats(e_b e_ci) nosubstat summstat(N) ctitle(Variable, Odds ratio, CI) replace 
//
tab part cond1 if ValidEmailYes1No0==1, col
tab part agecat if ValidEmailYes1No0==1, col
tab part gender if ValidEmailYes1No0==1, col
tab part ethn if ValidEmailYes1No0==1, col
tab part rel if ValidEmailYes1No0==1, col
tab part edu if ValidEmailYes1No0==1, col
********************************************************************************
// Comparison of study participants who chose the decoy 
tab agecat cond1 if ValidEmailYes1No0==1 & part==1, chi2  col
tab gender cond1 if ValidEmailYes1No0==1 & part==1, chi2 col
tab ethn cond1 if ValidEmailYes1No0==1 & part==1, chi2 exact col
tab rel cond1 if ValidEmailYes1No0==1 & part==1, chi2 exact col 
tab edu cond1 if ValidEmailYes1No0==1 & part==1, chi2 col
********************************************************************************
** Figure 2. Survey completing rate across experimental conditions
clear
use "C:\Users\sstoffel\Documents\temp_lit\decoy\exp.dta"
drop if ValidEmailYes1No0!=1
gen per=100*part
set scheme s2mono
collapse (mean) meanper= per (sd) sdper=per (count) n=per, by(cond1)
generate hiper = meanper + invttail(n-1,0.025)*(sdper / sqrt(n))
generate lowper = meanper - invttail(n-1,0.025)*(sdper / sqrt(n))
twoway (bar meanper cond1, barwidth(0.9)) (rcap hiper lowper cond), title("Individuals completing survey", size(medium) color(black)) xlabel(0 "Control condition" 1 "Decoy condition") xtitle("N=203", margin(medium)) ylabel (0(10)70) ytitle("Percentage") ylabel(, angle(0)) legend(off) graphregion(color(white)) bgcolor(white)
** Figure 3. Survey completing rate within decoy condition (order effeect)
clear
use "C:\Users\sstoffel\Documents\temp_lit\decoy\exp.dta"
drop if ValidEmailYes1No0!=1
gen per=100*part
set scheme s2mono
collapse (mean) meanper= per (sd) sdper=per (count) n=per, by(cond2)
generate hiper = meanper + invttail(n-1,0.025)*(sdper / sqrt(n))
generate lowper = meanper - invttail(n-1,0.025)*(sdper / sqrt(n))
twoway (bar meanper cond2, barwidth(0.9)) (rcap hiper lowper cond), title("Individuals completing survey", size(medium) color(black)) xlabel(0 "Control condition" 1 "Target shown first" 2 "Decoy shown first") xtitle("N=203", margin(medium)) ylabel (0(10)100) ytitle("Percentage") ylabel(, angle(0)) legend(off) graphregion(color(white)) bgcolor(white)
** Figure 3a. Preference for question types
clear 
input  id int1 int2 int3
    id	int1	int2	int3
  1.	61.4		7.9	30.7
end
lab define id 1 "Overall"
lab value id id
graph bar int1 int2 int3 ///
     , name(p1, replace) blabel(total)   ///
          ytitle(Percentages)   ///
          legend(lab(1 "Close-ended questions") lab(2 "Open-ended questions") lab(3 "Indifferent") lab() symxsize(5) size(small) row(1)) 
		  ** Figure 3b. Preference for question types
clear 
input  id int1 int2 int3
    id	int1	int2	int3
  1.	55.0		12.0	33.0
  2.	67.7		3.9		28.4
end
lab define id 1 "Control condition" 2 "Decoy condition"
lab value id id
graph bar int1 int2 int3 ///
     , name(p1, replace) over(id) blabel(total)   ///
          ytitle(Percentages)   ///
          legend(lab(1 "Close-ended questions") lab(2 "Open-ended questions") lab(3 "Indifferent") lab() symxsize(5) size(small) row(1)) 
** Figure 4. Attitudes towards late payment 
clear 
input  id int1 int2 int3 int4

    id	int1	int2	int3	int4
  1.	1.0		85.1	1.5		12.4
end
lab define id 1 "Overall"
lab value id id
graph bar int1 int2 int3 int4 ///
     , name(p1, replace) blabel(total)   ///
          ytitle(Percentages)   ///
          legend(lab(1 "Definitely not") lab(2 "Probably not") lab(3 "Probably yes") lab(4 "Definitely yes") symxsize(5) size(small) row(1)) 
** Figure 5. Attitudes towards late payment 
clear 
input  id int1 int2 int3 int4

            id       int1       int2       int3       int4
  1.            1 2.0 78.0 1.0 19.0
  2.            2 0.0 92.2 2.0 5.9
end
lab define id 1 "Control condition" 2 "Decoy condition"
lab value id id
graph bar int1 int2 int3 int4 ///
     , name(p1, replace) over(id) blabel(total)   ///
          ytitle(Percentages)   ///
          legend(lab(1 "Definitely not") lab(2 "Probably not") lab(3 "Probably yes") lab(4 "Definitely yes") symxsize(5) size(small) row(1)) 
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
* PRELIMINARY SURVEY
** Figure 2. Attitudes towards late payment   
clear 
input  id int1 int2 int3
    id	int1	int2	int3
  1.	60.0		7.6	32.4
end
lab define id 1 "Overall"
lab value id id
graph bar int1 int2 int3 ///
     , name(p1, replace) blabel(total)   ///
          ytitle(Percentages)   ///
          legend(lab(1 "Close-ended questions") lab(2 "Open-ended questions") lab(3 "Indifferent") lab() symxsize(5) size(small) row(1)) 
** Figure 3.  Attitudes towards late payment 
clear 
input  id int1 int2 int3 int4

    id	int1	int2	int3	int4
  1.	1.0		85.2	1.9		11.9
end
lab define id 1 "Overall"
lab value id id
graph bar int1 int2 int3 int4 ///
     , name(p1, replace) blabel(total)   ///
          ytitle(Percentages)   ///
          legend(lab(1 "Definitely not") lab(2 "Probably not") lab(3 "Probably yes") lab(4 "Definitely yes") symxsize(5) size(small) row(1)) 
