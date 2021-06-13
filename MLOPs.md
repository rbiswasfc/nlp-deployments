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


## LINUX Commands
### Curl
* Curl command allows us to query urls from the command line
* In addition to making typical get requests, we can post form data, authenticate users, save responses to file in our system
* Particularly useful when testing REST APIs (CRUD Operations)
* Examples
    * `curl https://rbiswasfc.github.io/`
        * This will fetch the html response from this site and display on screen
    * `curl -i https://rbiswasfc.github.io/`
        * This will include the response header as well. The `-i` flag stands for --include
    * `curl -d "first=Raja&last=Biswas" https://rbiswasfc.github.io/`
        * This submits a POST requests with data specified by the -d flag 
    * `curl -X PUT -d "first=Rana&last=Biswas" https://rbiswasfc.github.io/`
        * This will send a PUT request to the site server 

    * `curl -u username:password https://github.com/rbiswasfc/`
        * This can be used to authenticate a user to a server
    * `curl -o tmp.html https://rbiswasfc.github.io/`
        * This will download the response contents to tmp.html file. The `-o` flag here stands for output

### htop
[Reference](https://www.deonsworld.co.za/2012/12/20/understanding-and-using-htop-monitor-system-resources/)

`htop` is an interactive and real time process monitoring application which will show you your usage per cpu/core, as well as a meaningful text graph of your memory and swap usage.

* To install htop run the following
    * `brew install htop`
* CPU usage color bar [Defaults]
    * blue - low priority processes (nice >0)
    * green - normal (user) processes
    * red - kernel processes
    * cyan - virtualliz
* Memory color code bar
    * green - used
    * blur - buffer
    * yellow - cache
* Process stats
    * R - running
    * S - sleeping
    * Z - zombie
    * T - traced/stopped
    * D - disk sleep
* Load average
    *  The system load is a measure of the amount of computational work that a computer system performs
    * For example,  4.0 on a quad core represents 100% utilization. Anything under a 4.0 load average for a quad-core is ok as the load is distributed over the 4 cores.
    * The first number is a 1 minute load average, second is 5 minutes load average and the third is 15 minutes load average.
* Uptime
    * Uptime shows how long the system has been running


* NI
    * The nice value of the process, from 19 (low priority) to -20 (high priority). 
    * A high value means, the process is being nice, letting other have relative high priority. The OS permission restrictions for adjusting priority apply.
    * A niceness level increase by 1 should yield a 10% more CPU time to the process
* PRI
    * Kernel's internal priority of the process, usually just its nice value plus 20. Different for real time processes.
    *  Priorities range from 0 to 139 and the range from 0 to 99 is real time and 100 to 139 for users. (TODO: check)
* VIRT
    * The size of virtual memory for the process
    * It is the sum of memory it is actually using, memory it has mapped into itself e.g. 
        * The video card's RAM for the X server
        * files on disk that have been mapped into it e.g. shared library
        * memory shared with other processes
* RES
    * The resident set size of the process (text + data + stack). The size of processes used physical memory.
    * This also corresponds directly to the %MEM column
* SHR
    * The size of processes shared pages

* Usage
    * Scroll the process list horizontally and vertically using the arrow keys
    * Kill a process by pressing the F9 key
    * Re-nice a process by pressing the F7 or F8 keys
    * List open files used by a process by pressing the `l` key
    * Display only processes of a single user by pressing the `u` key
    * Display processes sorted by any htop column by pressing the F6 key
    * Display processes in a tree view by pressing the F5 key


