import pandas as pd
import json
from datetime import datetime
import openai
import dotenv
from pydantic import BaseModel
from typing import List
import random
import argparse

# Create response format
class Response(BaseModel):
    occupations: List[str]
    hobbies_interests: List[List[str]]

#Functions for generating GPT strings from ANES 
def format_list(words):
    if len(words) == 0:
        return ""
    elif len(words) == 1:
        return words[0]
    else:
        formatted_words = ", ".join(words[:-1])
        return f"{formatted_words}, and {words[-1]}"
    
# This code fetches lines from the ANES, and returns as readable dict
def get_anes_rows(number_rows, ignore_love_hate=False, ignore_party_identity=False, ignore_voted2020=False):

    df = pd.read_csv('anes_timeseries_2020_csv_20220210.csv', low_memory=False)

    df1 = df
    # V202545 how often post polituical content on twitter?
    # V202544 how often do you use twitter? 

    ##  Rename columns
    col_recode = {'V201600':'gender',
    'V203000': 'state',
    'V201511x': 'education',
    'V201534x':'employed',
    'V201549x':'race',
    'V201601':'sexOrientation',
    'V201602':'justifiedViolence',
    'V201617x': 'income',
    'V201627': 'selfCensor',
    'V201628':'gunsOwned',
    'V201005':'attentionPolitics',
    'V202073':'vote2020',
    'V201103':'vote2016',
    'V201105':'vote2012',
    'V201116':'afraid',
    'V201117':'outraged',
    'V201118': 'angry',
    'V201119':'happy',
    'V201120':'worried',
    'V201121':'proud',
    'V201122':'irritated',
    'V201123':'nervous',
    'V201201':'liberalOrConservative',
    'V201228':'partyIdentif',
    'V201231x':'strongIdentif',
    'V201232':'partyIdentity',
    'V201156':'feelingDemocratic',
    'V201157':'feelingRepublican',
    'V202544':'howOftenUseTwitter',
    'V201508':'martialStatus',
    'V201200':'liberalConservative',
    'V201529': 'occupation',
    'V202205y1': 'problem1',
    'V202205y2': 'problem2',
    'V202205y3': 'problem3'
    }

    df1 = df1.rename(columns=col_recode)

    #Copy for use later
    df1['V201231x'] = df1['strongIdentif'] 

    ## Recode values
    dic = {-9: None,
           1: 'male',
           2:'female'}

    df1 = df1.replace({'gender': dic})


    dic = {1: 'Alabama',
    2: 'Alaska',
    4: 'Arizona',   
    5: 'Arkansas',
    6: 'California',
    8: 'Colorado',
    9: 'Connecticut',
    10: 'Delaware',
    11: 'Washington DC',
    12: 'Florida',
    13: 'Georgia',
    15: 'Hawaii',
    16: 'Idaho',
    17: 'Illinois',
    18: 'Indiana',
    19: 'Iowa',
    20: 'Kansas',
    21: 'Kentucky',
    22: 'Louisiana',
    23: 'Maine',
    24: 'Maryland',
    25: 'Massachusetts',
    26: 'Michigan',
    27: 'Minnesota',
    28: 'Mississippi',
    29: 'Missouri',
    30: 'Montana',
    31: 'Nebraska',
    32: 'Nevada',
    33: 'New Hampshire',
    34: 'New Jersey',
    35: 'New Mexico',
    36: 'New York',
    37: 'North Carolina',
    38: 'North Dakota',
    39: 'Ohio',
    40: 'Oklahoma',
    41: 'Oregon',
    42: 'Pennsylvania',
    44: 'Rhode Island',
    45: 'South Carolina',
    46: 'South Dakota',
    47: 'Tennessee',
    48: 'Texas',
    49: 'Utah',
    50: 'Vermont',
    51: 'Virginia',
    53: 'Washington',
    54: 'West Virginia',
    55: 'Wisconsin',
    56: 'Wyoming'}

    df1 = df1.replace({'state': dic})


    dic = {-9: None,
           -8: None,
           -2: None,
          1: "Less than high school",
          2: "High school",
          3: "High school",
          4: "Bachelor’s degree",
          5: "Graduate degree"}

    df1 = df1.replace({'education': dic})


    dic = {-2: None,
           1: 'employed',
           2: 'unemployed',
           4: 'unemployed',
           5: 'unemployed',
           6: 'unemployed',
           7: 'unemployed',
           8: 'unemployed'} 

    df1 = df1.replace({'employed': dic})

    dic = {1: 'White',
           2: 'Black',
           3: 'Hispanic',
           4: 'Asian',
           5: 'Native American',
           6: 'Multiple races',
          -9: None,
           -8: None} 

    df1 = df1.replace({'race': dic})

    dic = {1: 'heterosexual',
           2: 'homosexual',
           3: 'bisexual',
           4: None,
          -9: None,
           -5: None} 

    df1 = df1.replace({'sexOrientation': dic})

    dic = {-9: None,
           -5: None,
          1:'I never or rarely stop myself from saying something because I think someone might call me a racist, a sexist, or otherwise a bad person',
          2: 'I never or rarely stop myself from saying something because I think someone might call me a racist, a sexist, or otherwise a bad person',
           3: 'I occasionally stop myself from saying something because I think someone might call me a racist, a sexist, or otherwise a bad person',
           4: 'I often stop myself from saying something because I think someone might call me a racist, a sexist, or otherwise a bad person',
           5: 'I often stop myself from saying something because I think someone might call me a racist, a sexist, or otherwise a bad person'} 

    df1 = df1.replace({'selfCensor': dic})


    dic = {-9: 0,-5: 0}

    df1 = df1.replace({'gunsOwned': dic})

    dic = {-9: None,
           1: "I always or most of the time pay attention to what's going on in government and politics",
           2: "I always or most of the time pay attention to what's going on in government and politics",
           3: "I pay attention to what's going on in government and politics about half the time",
           4: "I pay attention to what's going on in government and politics never or some of the time",
           5: "I pay attention to what's going on in government and politics never or some of the time"}

    df1 = df1.replace({'attentionPolitics': dic})


    dic = {-9: None,
           -8: None,
           -7: None,
           -1: "Didn't vote",
           1: 'Joe Biden',
           2: 'Donald Trump',
           3: None,
           4: None,
           5: None,
           -6: None,
           7: 'Donald Trump',
           8: None,
           11: None,
           12: None,}

    df1 = df1.replace({'vote2020': dic})


    dic = {-9: None,
           -8:None,
           -1:'Didn\t vote',
           1: 'Hillary Clinton',
           2: 'Donald Trump',
           5:  None}

    df1 = df1.replace({'vote2016': dic})

    dic = {-9: None,
           -8:None,
           -1:'Didn\t vote',
           1: 'Barack Obama',
           2: 'Mitt Romney',
           5:  None}

    df1 = df1.replace({'vote2012': dic})


    dic = {-9: None,
           -8: None,
           -4: None,
           -1: None,
           1: 'I consider myself a liberal',
           2: 'I consider myself a conservative',
           3: None}

    df1 = df1.replace({'liberalOrConservative': dic})

    df1['party'] = df1['partyIdentif']
    dic = {-9: 'Not sure',
           -8: 'Not sure',
           -4: 'Not sure',
           0: 'Not sure',
           1: 'Democrat',
           2: 'Republican',
           3: 'Independent',
          5: 'Not sure'}

    df1 = df1.replace({'party': dic})

    dic = {-9: None,
           -8: None,
           -4: None,
           0: None,
           1: 'Democrat',
           2: 'Republican',
           3: 'Independent',
          5: None}

    df1 = df1.replace({'partyIdentif': dic})

    dic = {-9: None,
           -8: None,
           1: 'Strong Democrat',
           2: 'Democrat',
           3: 'Independent who leans Democrat',
           4: 'Independent',
          5: 'Independent who leans Republican',
          6: 'Republican',
          7: 'Strong Republican'}

    df1 = df1.replace({'strongIdentif': dic})

    dic = {-9: None,
           -8: None,
           -1: None,
           1: 'My party is very important to my identity',
           2: 'My party is very important to my identity',
           3: 'My party is moderately important to my identity',
           4: 'My party is not important to my identity',
           5: 'My party is not important to my identity'}

    df1 = df1.replace({'partyIdentity': dic})

    dic = {-9: None,
           -8: None,
           1: 'married',
           2: 'married',
           3: 'widowed',
           4: 'divorced',
           5: 'separated',
           6: 'never married'}
    
    df1 = df1.replace({'martialStatus': dic})

    dic = {-9: None,
              -8: None,
              1: 'extremely liberal',
              2: 'liberal',
              3: 'slightly liberal',
              4: None,
                5: 'slightly conservative',
                6: 'conservative',
                7: 'extremely conservative',
                99: None
    }

    df1 = df1.replace({'liberalConservative': dic})

    dic = {
        -9: None,
        -1: None,
        1: "For-profit company or organization",
        2: "Non-profit organization (including tax-exempt and charitable organizations)",
        3: "Local government (for example: city or county school district)",
        4: "State government (including state colleges/universities)",
        5: "Active duty U.S. Armed Forces or Commissioned Corps",
        6: "Federal government civilian employee",
        7: "Owner of non-incorporated business, professional practice, or farm",
        8: "Owner of incorporated business, professional practice, or farm",
        9: "Worked without pay in a for-profit family business or farm for 15 hours or more per week"
    }

    df1 = df1.replace({'occupation': dic})

    dic  = {
    1: "Defense spending",
    2: "Middle East",
    3: "Iraq",
    4: "War",
    5: "Terrorism",
    6: "Veterans",
    7: "National defense",
    8: "Foreign aid",
    9: "Foreign Trade",
    10: "Protection of US jobs",
    11: "Serbia /Balkans",
    12: "China",
    13: "International affairs",
    14: "Energy crisis",
    15: "Energy prices",
    16: "Energy",
    17: "Environment",
    18: "Natural Resources",
    19: "Education and training",
    20: "School funding",
    21: "Education",
    22: "AIDS",
    23: "Medicare",
    24: "Health",
    25: "Welfare",
    26: "Poverty",
    27: "Employment",
    28: "Housing",
    29: "Social security",
    30: "Income",
    31: "Crime",
    32: "Race relations",
    33: "Illegal drugs",
    34: "Police problems",
    35: "Guns",
    36: "Corporate Corruption",
    37: "Justice",
    38: "Budget",
    39: "Size of government",
    40: "Taxes",
    41: "Immigration",
    42: "Campaign finance",
    43: "Political corruption",
    44: "Ethics",
    45: "Government power",
    46: "Budget priorities",
    47: "Partisan politics",
    48: "Politicians",
    49: "Government",
    50: "The economy",
    51: "Stock market",
    52: "Economic inequality",
    53: "Recession",
    54: "Inflation",
    55: "Economics",
    56: "Agriculture",
    57: "Science",
    58: "Commerce",
    59: "Transportation",
    60: "Community development",
    61: "Abortion",
    62: "Child care",
    63: "Overpopulation",
    64: "Public morality",
    65: "Domestic violence",
    66: "Family",
    67: "Young people",
    68: "Sexual identity /LGBT+ issues",
    69: "The media",
    75: "Sexism /Gender issues",
    76: "Afghanistan",
    77: "Syria",
    78: "Elections",
    79: "Religion",
    80: "Civility",
    81: "Unity /division",
    82: "Health care",
    700: None,
    750: None,
    800: None,
    990: None,
    997: None,
}

    df1 = df1.replace({'problem1': dic})
    df1 = df1.replace({'problem2': dic})
    df1 = df1.replace({'problem3': dic})



    # We select only people who ever use twitter
    # df1 = df1.loc[df1['howOftenUseTwitter'].isin((1,2,3,4,5,6))]

    # We select only people with age
    df1 = df1.loc[df1['V201507x']>=0]
    
    #Remove the very small nr of people who did not answer to political affiliation
    # df1 = df1.loc[df1['V201231x']>0]
    
    #Calculate partisanship    
    # Remove the small number of individuals who did not respond to partisan feeling temp
    df1 = df1.loc[(df1['feelingDemocratic']>=0) & (df1['feelingDemocratic']<=100) ]
    df1 = df1.loc[(df1['feelingRepublican']>=0) & (df1['feelingRepublican']<=100) ] 
    # We use the temperature responses for party, to focus on identity and get a -1,1 value for every individual
    df1['partisan'] = (df1['feelingRepublican'] - df1['feelingDemocratic'])/100
    
    ## Function that generates N random people, with WEIGHTING based on the ANES weighting    
    # Note that replacement is key here!
    random_rows = df1.sample(n=number_rows, replace=True) if number_rows is not None else df1
    
    random_dicts = random_rows.to_dict(orient="records")

    #These are the codes for media channels
    # V201634a yahoo.com
    # V201634b cnn.com
    # V201634c huffingpost.com 
    # V201634d nytimes.com
    # V201634e breitbart.com
    # V201634f foxnews.com
    # V201634g washingtonpost.com
    # V201634h theguardian.com
    # V201634i usatoday.com 
    # V201634j bbc.com 
    # V201634k npr.org
    # V201634m dailycaller.com
    # V201634n bloomberg.com
    # V201634p buzzfeed.com
    # V201634q nbcnews.com

    #Here we produce the persona description
    
    res = []
    for d in random_dicts:
        l = {}
        #media sources
        media = ['V201634a','V201634b','V201634c','V201634d','V201634e','V201634f','V201634g','V201634h','V201634i','V201634j','V201634k','V201634m','V201634n','V201634p']
        l['media'] = [m for m in media if d[m]==1]
        
        l['feelingDemocratic'] = d['feelingDemocratic']
        l['feelingRepublican'] = d['feelingRepublican']

        l['liberalConservative'] = d['liberalConservative']
        # Normalize the twitter use variable: 
        #NormalizeL: How many times per every second week?
        # 1. Many times every day: 50
        # 2. A few times every day: 30
        # 3. About once a day: 14
        # 4. A few times each week: 5
        # 5. About once a week: 2
        # 6. Once or twice a month: 1
        # l['howOftenUseTwitter'] = {1:50, 2:30, 3:14, 4: 5, 5:2, 6:1}[d['howOftenUseTwitter']]
        
        # In your spare time, you like to watch
        hobbies = {'V201631a':'American Idol','V201630r':'NCIS','V201631i':'Good Morning America','V201631r':'Saturday Night Live','V201632c':'Amor Eterno','V201633e':'The Dave Ramsey Show'}
        hobbies_liked = [v for k,v in hobbies.items() if d[k]==1]
        l['tvPrograms'] = hobbies_liked

        # V201631a PRE: MENTION: TV PROG - AMERICAN IDOL (ABC)
        # V201630r PRE: MENTION: TV PROG - NCIS (CBS)
        # V201631i PRE: MENTION: TV PROG - GOOD MORNING AMERICA (ABC)
        # V201631r PRE: MENTION: TV PROG - SATURDAY NIGHT LIVE (NBC)
        # V201632c PRE: MENTION: TV PROG - AMOR ETERNO
        # V201633e PRE: MENTION: RADIO PROG - THE DAVE RAMSEY SHOW

        l['partisan'] = d['partisan']
        
        temps = {
            'feelingDemocratic':'Democrats',
            'feelingRepublican':'Republicans',
            'V201151': 'Joe Biden',
            'V201152': 'Donald Trump',
            'V202168': 'Muslims',
            'V202169': 'Christians',
            'V202170': 'Jews',
            'V202171': 'Police',
            'V202172': 'transgender people',
            'V202173': 'scientists',
            'V202174': 'Black Lives Matter',
            'V202175': 'journalists',
            'V202178': 'NRA',
            'V202184': 'rural Americans',        
            'V202158': 'Anthony Fauci',
            'V202159': 'Christian Fundamentalists',
            'V202160': 'feminists',
            'V202161': 'liberals',
            'V202164': 'conservatives',
            'V202166': 'homosexuals',
            'V202176': 'NATO',
            'V202179': 'socialists',
            'V202180': 'capitalists',
            'V202183': '#MeToo'}
        
        lovelist = []
        hatelist = []
        for k,v in temps.items():
            if d[k] >= 0 and d[k] <= 10:
                hatelist.append(v)
            if d[k] <= 100 and d[k] >= 90:
                lovelist.append(v)

        l['loveList'] = lovelist
        l['hateList'] = hatelist
        
        #Create persona string
        l['persona'] = ""

        l['gender'] = d['gender']
        l['maritalStatus'] = d['martialStatus']
        l['income'] = d['income']
        l['age'] = d['V201507x']

        if d['gender'] is not None:
            l['persona'] += f"You are {d['gender']}.\n"

        if d['martialStatus'] is not None:
            l['persona'] += f"You are {d['martialStatus']}.\n"
            
        if d['income'] is not None and d['income']>0:
            if d['income'] >= 1 and d['income'] <= 10:
                incomeclass = 'low income'
            elif d['income'] >= 11 and d['income'] <= 20:
                incomeclass = 'middle income'
            else:
                incomeclass = 'high income'
            l['persona'] += f"You are {incomeclass}.\n"

        if d['V201507x'] is not None:
            l['persona'] += f"Age: {d['V201507x']}.\n" 
        
        religions = {1: 'Protestant', 2: 'Evangelical Protestant', 3: 'Black Protestant', 4: 'Protestant',  5: 'Catholic', 6: 'Christian', 7: 'Jewish', 9: 'not religious'}
        l['religion'] = d['V201458x']
        if d['V201458x'] in list(religions.keys()):
            l['persona'] += f"You are {religions[d['V201458x']]}.\n"

        l['state'] = d['state']
        l['education'] = d['education']
        l['race'] = d['race']
        l['sexOrientation'] = d['sexOrientation']

        if d['state'] is not None:
            l['persona'] += f"You are from {d['state']}.\n"
        if d['education'] is not None: 
            l['persona'] += f"Education: {d['education']}.\n"
        if d['race'] is not None: 
            l['persona'] += f"You are {d['race']}.\n"
        if d['sexOrientation'] is not None: 
            l['persona'] += f"You are {d['sexOrientation']}.\n"
        
        l['party'] = 'Democrat' if l['partisan'] < 0 else 'Republican' if l['partisan'] > 0 else 'Non-partisan'        
        
        #People who never talk about politics: we add other preferences
        if d['V202545'] == 5: # V202023
            l['persona'] += "You never talk about politics.\n" #
            l['never_talk_politics'] = True
        else:
            l['never_talk_politics'] = False
        
        #Fishing
        if d['V202567'] == 1:
            l['persona'] += "You like to go fishing or hunting.\n" #

        l['fishing'] = d['V202567']

        if not ignore_voted2020:
            if d['vote2020'] in ['Donald Trump','Joe Biden']:
                l['persona'] += f"You voted for {d['vote2020']} in 2020.\n"
                l['voted2020'] = True
                l['voted2020_for'] = d['vote2020']
            else:
                l['persona'] += "You didn't vote in 2020.\n"
                l['voted2020'] = False
        if not ignore_party_identity:
            #Generate party affiliation
            if l['partisan'] == 0:
                l['persona'] += 'You prefer neither political party.\n'
            if l['partisan'] < 0 and l['partisan'] > -0.2:
                l['persona'] += 'You prefer the Democrats.\n'
            if l['partisan'] > 0 and l['partisan'] < 0.2:
                l['persona'] += 'You prefer the Republicans.\n'
            if l['partisan'] <= -0.2 and l['partisan'] > -0.5:
                l['persona'] += 'You are a Democrat.\n'
            if l['partisan'] >= 0.2 and l['partisan'] < 0.5:
                l['persona'] += 'You are a Republican.\n'
            if l['partisan'] <= -0.5:
                l['persona'] += 'You are a strong Democrat.\n'
            if l['partisan'] >= 0.5:
                l['persona'] += 'You are a strong Republican.\n'

        l['justifiedViolence'] = d['justifiedViolence']
        
        if d['justifiedViolence'] in [3,4,5]:
            l['persona'] += "You think political violence is justified.\n"

        if not ignore_love_hate:
            if len(lovelist)>0:
                l['persona'] += f'You love {format_list(lovelist)}.\n'

            if len(hatelist)>0:
                l['persona'] += f'You hate {format_list(hatelist)}.\n'            


        l['arguePolitics'] =d['V202545']== 1 or d['V202545']== 2
        #You post online a lot or always about politics, or you get into political argument in the last 12 months
        if d['V202545']== 1 or d['V202545']== 2: #d['V202024']== 1 or 
            l['persona'] += "You like to argue about politics.\n"

            
        if d['liberalConservative'] is not None:
                l['persona'] += f'You consider yourself {d["liberalConservative"]}.\n' 

        problems = [d['problem1'],d['problem2'],d['problem3']]
        problems = [p for p in problems if p is not None and type(p) == str]

        if len(problems)==1:
            l['persona'] += f'You think the most important problem facing the country is {format_list(problems)}.\n'
        elif len(problems)>1:
            l['persona'] += f'You think the most important problems facing the country are {format_list(problems)}.\n'

        l['importantProblems'] = problems

        l['gunsOwned'] = d['gunsOwned']

        if d['gunsOwned'] > 0:
            l['persona'] += "You own guns.\n"


        if len(hobbies_liked)>0:
            l['persona'] += f'You like to watch {format_list(hobbies_liked)} on TV.\n'
        
        res.append(l)


    return res

def return_persona_string():
    
    personas = get_anes_rows(1)
    return personas[0]['persona']


def extend_with_ai(persona, client):

    prompt = f"""I am going to give you a persona of a person. I need you to fill in some other pieces, and generate the options THREE times:

- Your occupation
- A list of your hobbies / interests

Put them in lists

Please base it VERY LOOSELY on the persona. It does not need to align with the political stance and important problems.
Stay away from 'community' or 'volunteering'. Use your information on popular hobbies and occupations.
Please answer in the format I gave you. I will give you the persona now.

{persona['persona']}"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
        ],
        response_format=Response
    )

    response_class = response.choices[0].message.parsed

    occupations = response_class.occupations
    hobbies_interests = response_class.hobbies_interests

    chosen_occupation = random.choice(occupations)
    chosen_hobbies_interests = random.choice(hobbies_interests)

    persona['persona'] += f"Your occupation is {chosen_occupation}.\n"
    persona['persona'] += f"You like {format_list(chosen_hobbies_interests)}.\n"

def add_biography(persona, client, ignore_bio_love_hate=False, ignore_bio_party_identity=False, ignore_bio_voted2020=False):

    prompt = f"""Write a very short (max. 140 characters), very informal social media biography for the following persona: 

{persona['persona']}

You may add things that are not in the persona. Do not use emoji. Write as if you are the person described."""

    if ignore_bio_love_hate:
        prompt += "\nDo not include anything about things you love or hate."
    if ignore_bio_party_identity:
        prompt += "\nDo not include anything about your political party identity."
    if ignore_bio_voted2020:
        prompt += "\nDo not include anything about whether you voted in 2020."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
        ]
    )

    persona['biography'] = response.choices[0].message.content

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_personas", type=int, default=2000, help="Number of personas to generate")
    argparser.add_argument("--ignore_love_hate", action='store_true', default=False, help="Whether to ignore love/hate lists")
    argparser.add_argument("--ignore_party_identity", action='store_true', default=False, help="Whether to ignore party identity info")
    argparser.add_argument("--ignore_voted2020", action='store_true', default=False, help="Whether to ignore voted2020 info")
    # Here, the parameters control what is added to the public bio of the agent. The persona generation always uses all info.
    argparser.add_argument("--ignore_bio_love_hate", action='store_true', default=False, help="Whether to ignore love/hate lists in bio")
    argparser.add_argument("--ignore_bio_party_identity", action='store_true', default=False, help="Whether to ignore party identity info in bio")
    argparser.add_argument("--ignore_bio_voted2020", action='store_true', default=False, help="Whether to ignore voted2020 info in bio")
    args = argparser.parse_args()
    
    #Example usage
    # print(return_persona_string())

    personas = get_anes_rows(args.num_personas, 
                            ignore_love_hate=args.ignore_love_hate, 
                            ignore_party_identity=args.ignore_party_identity, 
                            ignore_voted2020=args.ignore_voted2020)
    
    personas_file_name = (
        f"{datetime.now().strftime('%Y%m%d')}_personas_{args.num_personas}_"
        f"{'noLoveHate_' if args.ignore_love_hate else ''}"
        f"{'noPartyId_' if args.ignore_party_identity else ''}"
        f"{'noVoted2020_' if args.ignore_voted2020 else ''}"
        f"{'noBioLoveHate_' if args.ignore_bio_love_hate else ''}"
        f"{'noBioPartyId_' if args.ignore_bio_party_identity else ''}"
        f"{'noBioVoted2020_' if args.ignore_bio_voted2020 else ''}"
        ".json"
    )
    json.dump(personas, open(personas_file_name,"w"))

    dotenv.load_dotenv('../.env')

    client = openai.OpenAI()

    personas = json.load(open(personas_file_name))
    i=1
    for persona in personas:
        print(i)
        i+=1
        extend_with_ai(persona, client)
        add_biography(persona, client)
        print(persona['biography'])
        print()

    filename = (
        f"{datetime.now().strftime('%Y%m%d')}_personas_with_bio_{args.num_personas}_"
        f"{'noLoveHate_' if args.ignore_love_hate else ''}"
        f"{'noPartyId_' if args.ignore_party_identity else ''}"
        f"{'noVoted2020_' if args.ignore_voted2020 else ''}"
        f"{'noBioLoveHate_' if args.ignore_bio_love_hate else ''}"
        f"{'noBioPartyId_' if args.ignore_bio_party_identity else ''}"
        f"{'noBioVoted2020_' if args.ignore_bio_voted2020 else ''}"
        ".json"
    )
    with open(filename, "w") as f:
        json.dump(personas, f)
