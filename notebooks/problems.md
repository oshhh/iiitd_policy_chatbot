# Problems
* "I am a student taught by instructor" vs "I taught by examples"
* The course will have 1.5 hours of lectures per week for the whole semester, or 3 hours of lectures per week for half the semester
* Imperfect coreference resolution
## which, who, if
* handle conditionals
```
# If a student is absent from the Institute for more than 20 days in a semester due to medical or any other reason , this may be converted to a semester leave for that semester by DOAA'
{
    'subject': 'this', 
    'relation': 'may be converted to', 
    'object': 'a semester leave', 
    'modifiers': [{'m_rel': 'for', 'm_obj': 'that semester'}, {'m_rel': 'by', 'm_obj': 'DOAA'}], 
    'subject_modifiers': [], 
    'condition': 'a student is absent from the Institute for more than 20 days in a semester due to medical or any other reason'
}
```
* handling which, who etc
```
# "Any problem shall be referred to the UG committee which may refer it to the Senate."
{
    'subject': 'Any problem', 
    'relation': 'shall be referred to', 
    'object': 'the UG committee', 
    'modifiers': [{'m_rel': 'which may refer', 'm_obj': 'it'}, {'m_rel': 'to', 'm_obj': 'the Senate'}], 'subject_modifiers': [], 
    'condition': None
}
```

* Increase in performance:
1 60 133 276 124 87 0.31521739130434784
1 60 133 276 147 104 0.37681159420289856


* omitted that, which etc : "student shall register for courses the student wishes to take"

## "modifier and modifier"
* "Any condition arising in the B.Tech. program and not covered in the regulations shall be referred to the UG committee"


## Imperfect sentence split
* Improvement
1 60 133 286 156 109 38.11
1 60 133 292 171 118 40.41
1 60 132 290 171 118 40.69

## remaining
For the remaining sentences extracted the entities in the sentences


## Chatbot output
### shortlist 30 sentences:
QUESTION: What is the normal load for UG students
ANSWER:
Normal load for the first and second year students is 16-20 credits and for the third and fourth year students is 16-22 credits.
Time taken: 21.39 secs

QUESTION: If I fail a course and take it again in the later semester, will my earlier course with F grade be removed from the transcript
ANSWER:
Also, all attempts in that course will be recorded in the respective semesters’ transcripts and the repeat course will not be permitted for late drop after mid-semester examination.
Time taken: 28.52 secs

QUESTION: what is the process of registration?
ANSWER:
Registration is compulsory for all students who are not on leave of absence, and is the sole responsibility of the student and must be completed before the last date of registration.
Time taken: 16.47 secs

QUESTION: how many seats are there in cse for admission?
ANSWER:
The number of seats and the process for application and admission are described in each year’s prospectus.
Time taken: 15.59 secs

QUESTION: what is the admission criteria for btech
ANSWER:
Admission to the undergraduate program at IIIT-Delhi is based on criteria which use scores from entrance test(s), class XII, and other competitions.
Time taken: 25.06 secs

QUESTION: I am in 1st year. Can I take overload?
ANSWER:
First year students cannot take any overload.
Time taken: 22.25 secs

QUESTION: I am in 2nd year. Can I take overload?
ANSWER:
First year students cannot take any overload.
Time taken: 24.4 secs

QUESTION: what happens if I miss the endsem because of a medical reason?
ANSWER:
For medical reasons, only two medical applications including mid-semester and end-semester would be accepted for make-up examinations during the entire duration of the btech degree completion.
Time taken: 15.1 secs

QUESTION: what happens if I fail a course?
ANSWER:
In case a student fails to do pre-registration of courses, a penalty may be imposed if a student does not register a student or a student preference within the stipulated period.
Time taken: 26.3 secs

QUESTION: what happens if I get an F grade in a course?
ANSWER:
An F and X grade obtained in any course shall be reflected in the grade sheet.
Time taken: 27.32 secs

### shortlisting 10 sentences:
QUESTION: What is the normal load for UG students
ANSWER:
Normal load for the first and second year students is 16-20 credits and for the third and fourth year students is 16-22 credits.
Time taken: 8.61 secs

QUESTION: If I fail a course and take it again in the later semester, will my earlier course with F grade be removed from the transcript
ANSWER:
The pass grade for the respective course will count towards the SGPA of the semester in which the course is passed and not in the semester where F’ grade was awarded.
Time taken: 10.96 secs

QUESTION: what is the process of registration?
ANSWER:
Registration is compulsory for all students who are not on leave of absence, and is the sole responsibility of the student and must be completed before the last date of registration.
Time taken: 9.41 secs

QUESTION: how many seats are there in cse for admission?
ANSWER:
The number of seats and the process for application and admission are described in each year’s prospectus.
Time taken: 8.79 secs

QUESTION: what is the admission criteria for btech
ANSWER:
Details about the admission criteria and entrance test(s) whose scores are to be used, are announced through the admission prospectus each year.
Time taken: 7.05 secs

QUESTION: I am in 1st year. Can I take overload?
ANSWER:
First year students cannot take any overload.
Time taken: 6.87 secs

QUESTION: I am in 2nd year. Can I take overload?
ANSWER:
First year students cannot take any overload.
Time taken: 7.82 secs

QUESTION: what happens if I miss the endsem because of a medical reason?
ANSWER:
For medical reasons, only two medical applications including mid-semester and end-semester would be accepted for make-up examinations during the entire duration of the btech degree completion.
Time taken: 9.62 secs

QUESTION: what happens if I fail a course?
ANSWER:
proper registration may be cancelled by the DOAA for a course if any irregularity is found at a later stage.
Time taken: 6.32 secs

QUESTION: what happens if I get an F grade in a course?
ANSWER:
An F and X grade obtained in any course shall be reflected in the grade sheet.
Time taken: 11.21 secs

