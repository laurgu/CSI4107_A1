package a1.lucene;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import main.java.a1.lucene.CustomAnalyzer;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.queryparser.classic.ParseException;

public class App {

    // Method to perform search
    private static void performSearch(Directory index, String queryString,
            CustomAnalyzer customAnalyzer, ArrayList<String> allResults, int queryNum)
            throws IOException, ParseException {
        // Open the index reader and searcher
        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);

        // Use custom analyzer and paser
        QueryParser queryParser = new QueryParser("processedText", customAnalyzer);

        String sanitizedQueryString = sanitizeQueryString(queryString);

        // Apply same prepreocessing to query string
        Query query = queryParser.parse(sanitizedQueryString);

        // Perform search and get top 1000 documents
        TopDocs topDocs = searcher.search(query, 1000);
        ScoreDoc[] scoreDocs = topDocs.scoreDocs;

        // Print query number and number of results

        int counter = 1;
        // Iterate over the top documents and store results
        for (ScoreDoc scoreDoc : scoreDocs) {
            int docId = scoreDoc.doc;
            float score = scoreDoc.score;

            Document doc = searcher.doc(docId);
            String docIdString = doc.get("docId");

            // Create result string and add to the results list
            String result = queryNum + " Q0 " + docIdString + " " + counter + " " + score + " " + "run_name\n";
            System.out.println(result);
            counter++;
            allResults.add(result);
        }
    }

    private static String sanitizeQueryString(String queryString) {
        String regex = "[-+|!{}\\[\\]^\"~*?:\\\\/]";
        // Replace special characters with a space
        String sanitizedQueryString = queryString.replaceAll(regex, " ");

        System.out.println(sanitizedQueryString);
        return sanitizedQueryString;
    }

    // Function reads queries from file and puts them into a list
    private static ArrayList<String> getQueries() {
        ArrayList<String> queries = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader("./Queries.txt"))) {
            String line;

            // Read each line from the file and add to the queries list
            while ((line = br.readLine()) != null) {
                queries.add(line);
            }

            return queries;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return queries;
    }

    // Main method to run the querrying
    public static void main(String[] args) throws IOException, ParseException {
        // Open the Lucene index
        Directory index = FSDirectory.open(Paths.get("./index_dir"));

        // Create a custom analyzer
        CustomAnalyzer customAnalyzer = new CustomAnalyzer();

        // Read queries from file
        ArrayList<String> queries = getQueries();

        // List to store all results
        ArrayList<String> allResults = new ArrayList<>();

        // Iterate over each query and perform search
        for (int i = 0; i < queries.size(); i++) {
            performSearch(index, queries.get(i), customAnalyzer, allResults, i + 1);
        }

        // Write results to a file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("Results.txt"))) {
            for (String result : allResults) {
                writer.write(result);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
