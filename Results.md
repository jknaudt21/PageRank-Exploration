# Pagerank Project
**Author:** Juan Diego Herrera

**Contributors:** Mike Izbicki

**Date:** November 20, 2020

This simple project explores a custom implementation of the Pagerank algorithm as well as the use of a personalization vector

This project was made as an assignment for a Data Mining course

## The Power Method

This implementation of Pagerank uses a "classic" power method. The only notable change is that we divide the pagerank vector by its norm. That way, we can reduce the number of iterations while at the same time remaining faithful with our accuracy

### Testing: `small.csv.gz`
For testing purposes, I first use a small 6-site graph. This small graph aids in ensuring that the output is correct without having to run a large data set. Below is the output using iPython: 
```
In [2]: run pagerank.py --data=./small.csv.gz
INFO:root:rank=0 pagerank=7.0150e-01 url=4
INFO:root:rank=1 pagerank=5.3858e-01 url=6
INFO:root:rank=2 pagerank=3.9632e-01 url=5
INFO:root:rank=3 pagerank=2.0443e-01 url=2
INFO:root:rank=4 pagerank=1.0243e-01 url=3
INFO:root:rank=5 pagerank=9.2101e-02 url=1
```
### Using Word2Vec to improve queries
This implementation of pagerank uses the `gensim` library to create word vectors and improve query results. Whenever a query (or personalization vector query) is made, we expand it by adding the 5 most similar words to it. This means that if we use the minus `-` sign in a query, we'd get the top urls the *exclude* the query and its 5 most similar words. For the `gensim` model I used `glove-twitter-25` simply because its lightweight, but the model could be improved if we used bigger `gensim` models. 

### Search Queries
The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then program returns all urls that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages on the blog related to our query.

**Note:** to ensure proper desired behavior, the search query must be entered using double (") quotes 

The examples below will now use the webgraph of [Lawfareblog.com](www.lawfareblog.com), from the file `lawfareblog.cvs.gz`, as its a larger and more relevant dataset

```
In [1]: run pagerank.py --data=./lawfareblog.csv.gz --search_query="weapons"
INFO:root:rank=0 pagerank=4.5717e-03 url=www.lawfareblog.com/why-did-you-wait-moral-emptiness-and-drone-strikes
INFO:root:rank=1 pagerank=3.1108e-03 url=www.lawfareblog.com/dc-district-court-dismisses-journalists-drone-lawsuit
INFO:root:rank=2 pagerank=2.0232e-03 url=www.lawfareblog.com/revived-cia-drone-strike-program-comments-new-policy
INFO:root:rank=3 pagerank=1.9668e-03 url=www.lawfareblog.com/us-court-appeals-dc-circuit-dismisses-suit-over-us-drone-strike
INFO:root:rank=4 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=5 pagerank=1.1620e-03 url=www.lawfareblog.com/slaughterbots-and-other-anticipated-autonomous-weapons-problems
INFO:root:rank=6 pagerank=1.1276e-03 url=www.lawfareblog.com/german-courts-weigh-legal-responsibility-us-drone-strikes
INFO:root:rank=7 pagerank=8.3740e-04 url=www.lawfareblog.com/shift-jsoc-drone-strikes-does-not-mean-cia-has-been-sidelined
INFO:root:rank=8 pagerank=7.8705e-04 url=www.lawfareblog.com/atomwaffen-division-member-pleads-guilty-firearms-charge
INFO:root:rank=9 pagerank=7.8571e-04 url=www.lawfareblog.com/waiving-imminent-threat-test-cia-drone-strikes-pakistan
```


```
In [3]: run pagerank.py --data=./lawfareblog.csv.gz --search_query="corona"
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9226e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0392e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9155e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7042e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6257e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5047e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3621e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1249e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0188e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

In [4]: run pagerank.py --data=./lawfareblog.csv.gz --search_query="snowden"
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=1 pagerank=2.1730e-03 url=www.lawfareblog.com/thoughts-assange-indictment-wheres-vault-7
INFO:root:rank=2 pagerank=2.1639e-03 url=www.lawfareblog.com/assange-indictment-seeks-punish-pure-publication
INFO:root:rank=3 pagerank=2.1639e-03 url=www.lawfareblog.com/us-media-crosshairs-new-assange-indictment
INFO:root:rank=4 pagerank=1.4839e-03 url=www.lawfareblog.com/justice-department-sues-edward-snowden
INFO:root:rank=5 pagerank=1.4821e-03 url=www.lawfareblog.com/lawfare-podcast-timothy-edgar-mass-surveillance-after-snowden
INFO:root:rank=6 pagerank=1.4710e-03 url=www.lawfareblog.com/fifth-anniversary-snowden-disclosures
INFO:root:rank=7 pagerank=1.4529e-03 url=www.lawfareblog.com/edward-snowden-national-security-whistleblowing-and-civil-disobedience
INFO:root:rank=8 pagerank=1.3622e-03 url=www.lawfareblog.com/assange-superseding-indictment-0
INFO:root:rank=9 pagerank=1.2001e-03 url=www.lawfareblog.com/israels-netanyahu-indicted-amid-political-gridlock
```
### Further Enhancements

**Filtering spam** 

In large graphs, the P matrix naturally contains a lot of structure.
In the case of Lawfare, most pages on the domain have links to the root page https://lawfareblog.com/  and similarly broad pages like https://www.lawfareblog.com/topics and https://www.lawfareblog.com/subscribe-lawfare.
These pages therefore have a large pagerank.
We can get a list of the pages with the largest pagerank by running:

```
In [5]: run pagerank.py --data=./lawfareblog.csv.gz
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
```

These pages are not very interesting because they are not articles. When a user is searching lawfare, they might not want Pagerank to suggest these pages. 

To find the most important articles we modify the P matrix by removing all links to non-article pages.

To overcome this hurdle, we use a method that removes pages with "too many" links. Of course, this is merely a heuristic and does not ensure full accuracy

The `--filter_ratio` argument removes all pages that have more links than the specified fraction.

Using this option, we can estimate the most important articles on the domain with the following command:
```
In [6]: run pagerank.py --data=./lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.5221e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9850e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9371e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5109e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5001e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=5 pagerank=1.4828e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.4828e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=7 pagerank=1.4689e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4304e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4189e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```

Notice that that we have removed the blog's most popular article (www.lawfareblog.com/snowden-revelations).

This only evinces that removing spam from Pagerank is an incredibly difficult task

**Tuning pagerank parameters**

Recall from the reading that the runtime of pagerank depends heavily on the eigengap of the $\bar\bar P$ matrix,
and that this eigengap is bounded by the alpha parameter.

We could run:
```
In [7]: run pagerank.py --data=./lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
This command would require a significantly higher amount of iterations and longer time. This implementation of Pagerank, for example, would require 690 iterations. 

Note that the amount of iterations is also increasing due to the `--filter_ratio` option, since Lawfareblog naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence.

As expected, changing the alpha gives us different results - especially when running with `--filter_ratio`

For example, 
```
In [8]: run pagerank.py --data=./lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=7.0150e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0150e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0558e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1754e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.1940e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6014e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6013e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6010e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6008e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6008e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

## The personalization vector

One of the greatest utilities in Pagerank is the fact that the algorithm allows for users to have unique results. This is achieved by using a *personalization vector*. The vector allows Pagerank to rank sites most relevant to the user if the sites url match a user's unique query.

In this implementation of Pagerank, we can enable the `--personalization_vector_query` command line argument,
which provides an alternative method for searching by doing the filtering on the personalization vector.

For example, we can consider a user that's fairly interested on the ongoing Coronavirus crisis. Then, their Pagerank results could look something like this: 
```
In [9]: run pagerank.py --data=./lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query="corona"
INFO:root:rank=0 pagerank=6.3213e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3211e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.4962e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.1626e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=4 pagerank=1.1626e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=5 pagerank=8.8833e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=8.5443e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=8.5443e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.1883e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=6.8968e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

Notice that these results differ than those from using `--search_query`

With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.

**Other uses of personalization vector:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
In [10]: run pagerank.py --data=./lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query="corona" --search_query="-corona"
INFO:root:rank=0 pagerank=6.3213e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3211e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.4962e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=8.8833e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=6.8562e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=5 pagerank=6.5838e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=6 pagerank=6.1389e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.5939e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.4060e-02 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=9 pagerank=4.9363e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
```
You can see that there are many urls about concepts that are obviously related like "covid", "clinical trials",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions. Plus, we're also omitting the words that `gensim` believes are most closely related to `corona`. Note that "covid" is included in these results because it is not a word in the `gensim` library

Another example of this advanced querying technique would be to look for articles related to the Snowden investigation but do not explicitly include `snowden` or similar words in the url

``` 
In [11]: run pagerank.py --data=./lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query="snowden" --search_query="-snowden"
INFO:root:rank=0 pagerank=4.2759e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.5901e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.5901e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=2.1517e-01 url=www.lawfareblog.com/senate-examines-threats-homeland
INFO:root:rank=4 pagerank=1.7603e-01 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
INFO:root:rank=5 pagerank=1.7603e-01 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
INFO:root:rank=6 pagerank=1.7593e-01 url=www.lawfareblog.com/whats-house-resolution-impeachment
INFO:root:rank=7 pagerank=1.6738e-01 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
INFO:root:rank=8 pagerank=1.6122e-01 url=www.lawfareblog.com/huawei-foreign-power-or-agent-foreign-power-under-fisa-insights-sanctions-case
INFO:root:rank=9 pagerank=1.5172e-01 url=www.lawfareblog.com/open-letter-gchq-threats-posed-ghost-proposal
```