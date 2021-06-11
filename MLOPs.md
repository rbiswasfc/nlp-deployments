This is notes from CS 329S: Machine Learning Systems Design course for my future reference.


ML in Research vs Production

Research:
* Model performance is the main objective
* Priority is fast training and high throughput
* Data is normally hold fixed and experiments are conducted to improve the model

Production:
* Different stakeholders have different objectives
* Priority is fast inference with low latency
* Data in the wild shifts constantly (data drift)
* Fairness and interpretability are important concerns


# Lecture 3

## Data vs Model
Two different philosophy in developing ML systems
* Improve the model (model-centric)
* Improve the data (data-centric)

"""
The debate isn’t about whether finite data is necessary, but whether it’s sufficient. The term finite here is important, because if we had infinite data, we can just look up the answer. Having a lot of data is different from having infinite data.
"""

## Data sources
* User generated data (requires fast processing)
    * Active
        * Clicks
        * Inputs
        * Likes
        * Comments
        * Share
    * Passive
        * Ignoring ads/pop-ups
        * Spending time in a page/video

* System generated data
    * Logs
    * Metadata (?)
    * Model predictions

* Enterprise app data [adobe data]

* First-party data: data collected by a company from their own user base [Adobe]
* Second-party data: data collected by another company from their own users [StarHub] 
* Third-party data: data collected by a company from general public who aren't their customer [Lotame]. Third party data are usually sold by vendors after cleaning and processing

Mobile Advertiser ID: unique id to aggregate data from all activities on phone

## Data Formats
The process of converting a data structure or object state into a format that can be stored or transmitted later is called data serialization.

Row based vs Column based storage
Text vs Binary

## OLTP vs OLAP 
ACID property in OLTP

DB vs EDW vs DataMart

OLTP databases are processed and aggregated to generate OLAP databases via ETL

## Structured vs unstructured data
ETL vs ELT

## Batch Processing vs Stream Processing

User-facing application requirements
* Fast inference  
* Real time processing <-> Stream processing
    * In-memory storage vs Permanent storage
    * Challenges
        * Unknown data size
        * Unknown data arrival frequency

 Having two different pipelines to process your data is a common cause for bugs in ML production
    * unknown feats
    * data transformation mismatch

## Training dataset creation
* Data bias
* Labelling
    * Instruction ambiguity
    * Consistency