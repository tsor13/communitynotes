from transformers import AutoTokenizer, AutoModelForCausalLM
import diskcache as dc

import requests
from bs4 import BeautifulSoup
import re
from googleapiclient.discovery import build
import pprint

import os
import json


def scrape_news_article(url):
    # Send a GET request to the URL
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code != 200:
        return "Failed to retrieve the article.", "Failed to retrieve the article."

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the title of the article
    title = soup.find('h1')  # Adjust the tag as per the structure of the website
    if title:
        title = title.get_text().strip()
    else:
        title = "Title not found"

    text = soup.get_text()
    # change all multiple newlines to single newline
    text = re.sub(r'\n+', '\n', text)
    # where there are many lines in a row with fewer than 4 words, remove those line
    lines = text.split('\n')
    lines = [line for line in lines if len(line.split()) > 4]
    # get average word length in line, if average word length is over 10, remove line
    lines = [line for line in lines if len(line) / len(line.split()) < 10]
    text = '\n'.join(lines)
    # remove everything after subscribe appears (cap insensitive), bridging newlines
    # text = re.sub(r'\n*subscribe.*+', '', text, flags=re.IGNORECASE)
    truncate_words = ['subscribe', 'Subscribe']
    for word in truncate_words:
        if word.lower() in text.lower():
            text = text.split(word)[0]

    return title, text

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def append_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + '\n')

my_api_key = ""
my_cse_id = ""

# Define a global cache with a maximum size limit
cache = dc.Cache('search_cache', size_limit=10**9)  # 1 GB limit

def google_search(search_term, api_key, cse_id, before_date, **kwargs):
    # Create a cache key based on the function's arguments
    cache_key = (search_term, cse_id, before_date, tuple(sorted(kwargs.items())))

    # Try to get cached results
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    # Perform the search if no cache is found
    service = build("customsearch", "v1", developerKey=api_key)
    sort_param = f"date:r:19000101:{before_date.replace('-', '')}"
    res = service.cse().list(q=search_term, cx=cse_id, sort=sort_param, **kwargs).execute()
    result_items = res.get('items', [])

    # Cache the results
    cache.set(cache_key, result_items)

    return result_items





class MisinfoDetector:
    def __init__(self, model_name='CohereForAI/c4ai-command-r-plus-4bit'):
    # def __init__(self, model_name='CohereForAI/c4ai-command-r-v01-4bit'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        self.model_name = model_name
    
    def extract_claims(self, text):
        # Extract claims from text
        conversation = [
            # {"role": "user", "content": "Here is a language model chat history: ```" + text + "```\n\nPlease enumerate the claims made in the most recent message."}
            {"role": "user", "content": '''I will give you a Tweet to fact-check and provide useful context for a viewer of the Tweet.''' + text + '''\nList all the claims and implications in the Tweet, or anything that a reader might believe after reading the Tweet.'''},
        ]
        input_ids = self.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        gen_tokens = self.model.generate(
            input_ids, 
            max_new_tokens=500, 
            do_sample=True, 
            temperature=0.3,
        )
        # gen_text = self.tokenizer.decode(gen_tokens[0])
        # decode only the new text
        gen_text = self.tokenizer.decode(gen_tokens[0][input_ids.shape[-1]:-1])
        return gen_text
    
    def make_search_terms(self, text, claims):
        # Make search terms from text and claims
        conversation = [
            {"role": "user", "content": '''I will give you a Tweet to fact-check and provide useful context for a viewer of the Tweet.''' + text + '''\nHere are some potential claims/implications made in the Tweet: ''' + claims + '''\nPlease provide search terms to verify the claims in a python list. You may make a flexible number of searches (anywhere from 2-10) depending on what is needed to verify the claims. Output only the python list, beginning and ending with brackets.'''},
        ]
        input_ids = self.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        gen_tokens = self.model.generate(
            input_ids, 
            max_new_tokens=500, 
            do_sample=True, 
            temperature=0.3,
        )
        gen_text = self.tokenizer.decode(gen_tokens[0])
        # print('Gen text:', gen_text)
        # get terms from the last message
        # if "=" in gen_text:
        #     search_term_str = gen_text.split('```')[-2].split('\n')[1:-1][0].split('=')[1].strip()
        # else:
        #     search_term_str = gen_text.strip()
        # remove everything before and up until [ and everything after and including ]
        search_term_str = gen_text.split('[')[-1].split(']')[0]
        # remove [ and ]
        search_terms = search_term_str[1:-1].split(',')
        # remove any leading or trailing apostrophes
        search_terms = [term.strip().strip('\'').strip('\"') for term in search_terms]
        return search_terms

    def get_documents_from_search_term(self, search_term, before_date):
        # Get documents from search term
        search_results = google_search(search_term, my_api_key, my_cse_id, before_date, num=5)
        # to the format expected by the model, which is a dictionary with keys 'title' and 'text'
        documents = [{'title': doc['title'], 'text': doc['snippet']} for doc in search_results]
        return search_results, documents
    
    def get_documents_from_search_terms(self, search_terms, before_date):
        # Get documents from search terms
        search_results = []
        documents = []
        for search_term in search_terms:
            # documents.extend(self.get_documents_from_search_term(search_term, before_date))
            search_result, document = self.get_documents_from_search_term(search_term, before_date)
            search_results.extend(search_result)
            documents.extend(document)
        return search_results, documents
    
    def get_docs_from_search_results(self, search_results):
        documents = []
        # try to scrape each link
        for search_result in search_results:
            url = search_result['link']
            print(url)
            title, text = scrape_news_article(url)
            if len(text) > 50:
                documents.append({'title': title + ' - ' + url , 'text': text[:min(4000, len(text))]})
                print('Added document')
            else:
                print('Document too short')
        return documents
    
    def correct_misinfo(self, text, claims, documents):
        conversation = [
            # {"role": "user", "content": "Here is a language model chat history: ```" + text + "```, along with the claims made in the most recent message: ```" + claims + "Based on the retrieved documents, please correct any misinformation or mistakes in the last message."}
            {"role": "user", "content": '''I will give you a Tweet to fact-check and provide useful context for a viewer of the Tweet.''' + text + '''\nHere are some potential claims/implications made in the Tweet: ''' + claims + '''\nPlease provide a 1-2 sentence statement to provide context/fact-check the claims in the tweet, directly addressing the author's claims.'''
            # + '''Only use the retrieved documents that appear reputable and relevant. Cite them specifically as needed.'''
            },
        ]
        input_ids = self.tokenizer.apply_grounded_generation_template(
            conversation,
            documents=documents,
            citation_mode="accurate", # or "fast"
            # citation_mode="fast", # or "accurate"
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to('cuda')
        print('Shape of input_ids:', input_ids.shape)
        gen_tokens = self.model.generate(
            input_ids, 
            max_new_tokens=500, 
            do_sample=True, 
            temperature=0.3,
        )

        gen_text = self.tokenizer.decode(gen_tokens[0][input_ids.shape[-1]:-1])
        # get only the grounded answer
        # split by newline
        print(gen_text)
        gen_text = gen_text.split('\n')
        # get the grounded answer
        gen_text = gen_text[-2]
        gen_text = ' '.join(gen_text.split(' ')[1:]).strip()
        return gen_text
    
    def quote_sources(self, text, claims, documents):
        conversation = [
            {"role": "user", "content": '''I will give you a Tweet to fact-check and provide useful context for a viewer of the Tweet.''' + text + '''\nHere are some potential claims/implications made in the Tweet: ''' + claims + '''\nPlease quote 2-6 snippets (from 1-5 sentences each) verbatim from the documents that are most relevant to fact-checking the tweet.'''},
        ]
        input_ids = self.tokenizer.apply_grounded_generation_template(
            conversation,
            documents=documents,
            citation_mode="accurate", # or "fast"
            # citation_mode="fast", # or "accurate"
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        gen_tokens = self.model.generate(
            input_ids, 
            max_new_tokens=1000, 
            do_sample=True, 
            temperature=0.3,
        )

        gen_text = self.tokenizer.decode(gen_tokens[0][input_ids.shape[-1]:-1])
        return gen_text

    
    def pipeline(self, text, date):
        outputs_directory = 'outputs'
        output_file = os.path.join(outputs_directory, 'pipeline_outputs.jsonl')
        ensure_directory(outputs_directory)

        # Initialize a dictionary to hold all outputs
        output_data = {'text': text, 'date': date, 'model_name': self.model_name}

        # Run the pipeline
        print('Text:', text)
        claims = self.extract_claims(text)
        print('Claims:', claims)
        output_data['claims'] = claims

        search_terms = self.make_search_terms(text, claims)
        print('Search terms:', search_terms)
        output_data['search_terms'] = search_terms

        search_results, documents = self.get_documents_from_search_terms(search_terms, date)
        output_data['search_results'] = search_results
        documents = self.get_docs_from_search_results(search_results)
        print('Number of documents:', len(documents))
        output_data['documents'] = documents

        correction = self.correct_misinfo(text, claims, documents)
        print('Correction:', correction)
        output_data['correction'] = correction

        quoted_sources = self.quote_sources(text, claims, documents)
        print('Quoted sources:', quoted_sources)
        output_data['quoted_sources'] = quoted_sources

        # Append each stage's data to a jsonl file
        append_to_jsonl(output_file, output_data)

        return correction



if __name__ == '__main__':
    text = '''Author: Monica Crowley (@MonicaCrowley)
Date: Aug 27, 2022
Tweet: They framed Trump as an “insurrectionist” who blocked the “legitimate and peaceful transfer of power” - as they walked away with their rigged election'''
    # example_convo = '''User: Where does Messi play football?
    # Assistant: Messi plays football at FC Barcelona.'''
    detector = MisinfoDetector()
    print(detector.pipeline(text, '2022-08-27'))
    # 35b
    # There is no evidence that the 2020 election was rigged and President Trump's claims of mass voter fraud have been debunked by independent fact-checking organisations and CNN. The transfer of power after the 2020 election was peaceful, despite a violent insurrection at the US Capitol incited by Trump's false allegations of a stolen election. 
    # 35B - quotes
    #     Answer: Here are some quotes relating to the claims made in Monica Crowley's tweet:
    # 
    # 1. The 2020 election was rigged:
    #  > " Mollie Hemingway established a clear distinction between fraud... and the focus of her book, which is systemic bias and corruption of the election system."
    #  > "Although Joe Biden won, it is the sole result of a rigged election."
    # 
    # 2. The transfer of power was not peaceful:
    #  > "A mob insurrection stoked by false claims of election fraud and promises of violent restoration."
    #  > "Extremists emboldened by President Trump had sought to thwart the peaceful transfer of power..."
    # 
    # 3. There is a double standard:
    #  > "When Black people protest for our lives, we are all too often met by National Guard troops or police equipped with assault rifles, shields, tear gas and battle helmets."
    #  > "When white people attempt a coup, they are met by an underwhelming number of law enforcement personnel who act powerless to intervene, going so far as to pose for selfies with terrorists."
    # 
    # 4. Trump was accused of insurrection:
    #  > "Senate votes to acquit Trump for incitement of Jan. 6 insurrection."
    #  > "Trump lawyers William J. Brennan and Michael van der Veen."
    # Grounded answer: Here are some quotes relating to the claims made in Monica Crowley's tweet:
    # 
    # 1. The 2020 election was rigged:
    #  > <co: 0>" Mollie Hemingway established a clear distinction between fraud... and the focus of her book, which is systemic bias and corruption of the election system."</co: 0>
    #  > <co: 0>"Although Joe Biden won, it is the sole result of a rigged election."</co: 0>
    # 
    # 2. The transfer of power was not peaceful:
    #  > <co: 6>"A mob insurrection stoked by false claims of election fraud and promises of violent restoration."</co: 6>
    #  > <co: 10>"Extremists emboldened by President Trump had sought to thwart the peaceful transfer of power..."</co: 10>
    # 
    # 3. There is a double standard:
    #  > <co: 11>"When Black people protest for our lives, we are all too often met by National Guard troops or police equipped with assault rifles, shields, tear gas and battle helmets."</co: 11>
    #  > <co: 11>"When white people attempt a coup, they are met by an underwhelming number of law enforcement personnel who act powerless to intervene, going so far as to pose for selfies with terrorists."</co: 11>
    # 
    # 4. Trump was accused of insurrection:
    #  > <co: 4>"Senate votes to acquit Trump for incitement of Jan. 6 insurrection."</co: 4>
    #  > <co: 4>"Trump lawyers William J. Brennan and Michael van der Veen."</co: 4>

    text = '''Author: Amazon News @amazonnews
Date: Mar 24, 2021
Tweet: Replying to @RepMarkPocan
1/2 You don’t really believe the peeing in bottles thing, do you? If that were true, nobody would work for us. The truth is that we have over a million incredible employees around the world who are proud of what they do, and have great wages and health care from day one.'''
    print(detector.pipeline(text, '2021-03-24'))
    # 35b
    # Several sources, including investigations by The Verge, The Guardian, and The Independent, confirm that Amazon employees are pressured to skip bathroom breaks and instead use bottles to pee into due to the proximity to toilets, fear of discipline for 'idle time', and demanding productivity quotas. However, Amazon has released statements disputing these claims and highlighting the benefits they provide to employees, including healthcare and competitive wages.

    #     Here is some further context provided by two sources regarding Amazon's response to such claims and the general working conditions: 
    # 
    # > Amazon ensures all of its associates have easy access to toilet facilities which are just a short walk from where they are working. [...] We have a focus on ensuring we provide a great environment for all our employees.
    # Source: *Amazon Official Spokesperson*, *The Observer*
    # 
    # > Starting on your first day of employment, Amazon offers a wide range of benefits to support you and your family at home and beyond. [...] Medical plans include coverage for prescription drugs, emergency and hospital care, mental health, X-rays, lab work, etc.
    # Source: *Amazon Benefits Overview*
    # Grounded answer: > <co: 3>Amazon workers pee into bottles to save time: investigator</co: 3>
    # > [...]
    # > <co: 3>Amazon warehouse staff are peeing in bottles because bathrooms are hundreds of yards away. One ex-worker said staffers fear they
    # ’ll get into trouble for taking too long away from the job.</co: 3>
    # > [...]
    # > <co: 3,7>Workers “lived in fear of being ­disciplined over ‘idle time’ and ­losing their jobs just because they needed the loo.</co: 3,7>”
    # Source: <co: 3>*The New York Post</co: 3>*
    # 
    # > <co: 1>Amazon warehouse workers skip bathroom breaks to keep their jobs, says report</co: 1> [...]
    # > <co: 1>In the UK, an undercover reporter and a labor survey exposed harrowing work conditions</co: 1> [...]
    # > <co: 1>74 percent of workers avoid using the toilet for fear of being warned they had missed their target numbers.</co: 1>
    # Source: <co: 1>*The Verge*</co: 1>
    # 
    # > <co: 5>'I'm not a robot': Amazon workers condemn unsafe, grueling conditions at warehouse</co: 5> [...]
    # > <co: 5>Employees under pressure to work faster call on retail giant to improve conditions</co: 5> [...]
    # > <co: 5>The petition called on Amazon to consolidate workers’ two 15-minute breaks into a 30-minute one. Workers say it can take up to 15 minutes just to walk to and from the warehouse break room.</co: 5>
    # Source: <co: 5>*The Guardian*</co: 5> 
    # 
    # These snippets provide evidence that Amazon employees have indeed been <co: 0,1,3,5,7>required to urinate in bottles while at work.</co: 0,1,3,5,7> 
    # 
    # Here is some further context provided by two sources regarding Amazon's response to such claims and the general working conditions: 
    # 
    # > <co: 3,7>Amazon ensures all of its associates have easy access to toilet facilities which are just a short walk from where they are working.</co: 3,7> [...] <co: 1,3



    text = '''Author: Jory Micah #ResistTyranny (@jorymicah)
Date: Apr 15, 2024
Tweet: Hitler was probably a Rothschild. He probably did not kill himself and probably did live the rest of his live out in Argentina.
Again, Rothschilds funded all sides of all world wars.

We all got played!!!'''
    print(detector.pipeline(text, '2024-04-15'))
    # 7B
    #There is no evidence that Hitler was a Rothschild, and historical accounts, along with dental records, unanimously describe Hitler's death in 1945 by suicide in Berlin, contradicting claims of him fleeing to Argentina.
    #
    #     1. Claim: Hitler was a Rothschild.
    # 
    # > Hitler's grandmother was a servant maid in the Rothschild mansion in Vienna, and his mother was the illegitimate daughter of a Rothschild. ([Source](https://trove.nla.gov.au/newspaper/article/98241616))
    # 
    # 2. Implication: The Rothschild family controlled Hitler and manipulated World War II.
    # 
    # > The widely held view that [Hitler] shot himself in his Berlin bunker on 30 April 1945 is challenged by a theory that Hitler escaped to Argentina...The Escape of Adolf Hitler, set out the case for a scenario almost too horrible to contemplate: that the Führer and Eva Braun made a home in the foothills of the Andes and had two daughters. ([Source](https://www.theguardian.com/world/2013/oct/27/hitler-lived-1962-argentina-plagiarism))
    # 
    # 3. Claim: Hitler did not kill himself but fled to Argentina.
    # 
    # > The notorious claim that Hitler escaped his Berlin bunker to live incognito in Argentina first gained popular currency in 1945, when Stalin spoke of it. ([Source](https://www.theguardian.com/world/2013/oct/27/hitler-lived-1962-argentina-plagiarism))
    # 
    # > Most conspiracy theories hold that Hitler and his wife, Eva Braun, survived and escaped from Berlin, with some asserting that he went to South America. ([Source](https://en.wikipedia.org/wiki/Conspiracy_theories_about_Adolf_Hitler%27s_death))
    # 
    # > A 2018 analysis of Hitler's teeth and bones concluded that he died in 1945 after taking cyanide and shooting himself in the head, putting to rest theories of his escape to Argentina. ([Source](https://www.history.com/news/hitler-death-cause-teeth-analysis))
    # 
    # 4. Implication: The accepted historical narrative of Hitler's death is false.
    # 
    # > Conspiracy theories about the death of Hitler contradict the accepted fact that he committed suicide... Stemming from a campaign of Soviet disinformation, most of these theories hold that Hitler survived and escaped. ([Source](https://en.wikipedia.org/wiki/Conspiracy_theories_about_Adolf_Hitler%27s_death))
    # 
    # 5. The Rothschild family funded all sides of World War II.
    # 
    # > I couldn't find any substantial evidence of this in my search.
    # 
    # 6. Belief: The public has been deceived about World War II. 
    # 
    # > Eyewitnesses and Hitler's dental remains demonstrate that he died in his Berlin bunker in 1945... [The] consensus of historians [is] that Hitler killed himself. ([Source](https://en.wikipedia.org/wiki/Conspiracy_theories_about_Adolf_Hitler%27s_death))
    # Grounded answer: Here are some quotes that are relevant to the claims made in the Tweet you provided. 

    text = '''Author: Kian (@naturelvr49)
Date: Apr 15, 2024
Tweet: Whales are not actually mammals. If Humans (land mammals) can't drink seawater — just try it! — how can supposed sea mammals like whales stay hydrated?'''
    print(detector.pipeline(text, '2024-04-15'))
    # 35B
    # While it's true that whales are mammals, their biology is very different from land mammals like humans, and they do not drink seawater voluntarily. Whales are thought to meet their hydration needs by consuming metabolic water, which is created when they break down food into nutrients, and then into energy and water. They also have specialized kidneys that allow them to process and excrete salt efficiently.

    # Quotes:
    # On the scientific classification of whales as mammals:
    # > Whales are a member of the cetacean family, and as such, despite being wholly water-resident, whales are mammals, not fish.           
    # > Whales are fully aquatic, placental marine mammals. [...] Whales, dolphins and porpoises belong to the order Cetartiodactyla, which co$sists of even-toed ungulates.
    # 
    # On the drinkability of seawater and whale hydration:
    # > Baleen whales do swallow a bit of water when they feed because they engulf large quantities of food (krill or fish) at a time and end $p ingesting some seawater in the process. How could they consume seawater and not get dehydrated like us? Short answer: they do not drin$ it voluntarily and they can filter seawater salt efficiently.
    # > Whales are capable of drinking seawater because they have specialized kidneys to process the salt, which is excreted in their urine. E$en though they can drink salt water, whales are thought to get the bulk of the water they need from their prey [...] As the whale proces$es the prey, it extracts water.
    # > The kidneys of marine mammals are so effective that the latter are capable of excreting a urine that has a higher salt concentration t$an the sea water itself and are thus able to access a supply of fresh water by ingesting salt water.                                    


