# Information Retrieval System

This project is an infomation retrieval system built using Apache Lucene.

## Group Information

**Students:**

- Ishan Phadte 300238878
- Lauren Gu 300320106
- Angus Leung 300110509

**Division of Work:**

- Ishan Phadte: Part 1 & Part 2
- Lauren Gu: Part 3
- Angus Leung: Part 2

## Prerequisites

- Java Development Kit (JDK)

## Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/laurgu/CSI4107_A1.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd A1-lucene
   ```

3. **Build the project using gradle wrapper:**

   ```bash
   ./gradlew build
   ```

   **_Or using gradle if installed:_** `gradle build`

4. **Build the index:**

   ```bash
   ./gradlew runLuceneIndex
   ```

   **_Or if gradle installed:_** `gradle runLuceneIndex`

5. **Run the the search function using gradle wrapper:**

   ```bash
   ./gradlew run
   ```

   **_Or using gradle if installed:_** `gradle run`

6. **To view results, navigate to app directory then open the Results.txt file:**

   ```bash
   cd app

   notepad Results.txt
   ```

## Functionality

### Part 1

1. **Jsoup Parsing:** The files in the coll folder are read using Jsoup which allows ill-formatted xml files to still be parsed.

2. **Document Extraction:** The seperate documents in each file are extracted by identifying \<DOC> tags. The \<DOCNO> and \<TEXT> of each document are extracted.

3. **Preprocessing:** The documents are preprocessed using a custom analyzer that extends the analyzer class provided by Lucene.
   It tokenizes words and applies lowercasing and stop word removal.

### Part 2

The index is created using Lucene's "Index Writer". This index is written to a file where it can be reused for different queries.

### Part 3

1. The query string is preprocessed using the custome analyzer, like how documents are preprocessed in Part 1.

2. Querying is done using Lucene's "Index Searcher" to search the index build in Part 2. It retrieves the top 1000 results for a given query and these results are written to a txt file.

### Optimizations

Initially, we implemented our IR system using the tf-idf weighting system. For comparison we implemented this Lucene version and found it seemed to produce more accurate results without significantly impacting the runtime.

### Data Structures 

A Set was used for stops words because each stop word is unique and order isn't required

A Dictionary was used for the inverted index because we needed a key value data structure and a dictionary fits the description 


### Sample of 100 Tokens 

['nation', 'governors', 'appealed', 'whitehouse', 'sunday', 'relief', '163', 'federal', 'rules', 'regulations', 'andheard', 'former', 'governor', 'call', 'constitutional', 'convention', 'torestore', 'states', 'rights', 'new', 'hampshire', 'gov', 'john', 'h', 'sununu', 'opening', 'nationalgovernors', 'association', 'winter', 'meeting', 'said', 'time', 'hascome', 'press', 'new', 'division', 'authority', 'statesand', 'washington', 'erosion', 'fundamental', 'balance', 'struck200', 'years', 'ago', 'philadelphia', 'sununu', 'nga', 'chairman', 'said', 'ata', 'news', 'conference', 'gaveling', 'first', 'plenary', 'session', 'toorder', 'president', 'reagan', 'black', 'tie', 'dinner', 'governors', 'sundaynight', 'told', 'governors', 'envied', 'balanced', 'budgetrequirements', 'line', 'item', 'vetoes', 'many', 'possess', 'notone', 'would', 'put', 'mess', 'inwashington', 'budget', 'time', 'president', 'said', 'also', 'said', 'want', 'tie', 'successor', 'hands', 'butexpressed', 'hope', 'next', 'president', 'would', 'continue', 'tradition', 'ofinviting', 'governors', 'white']