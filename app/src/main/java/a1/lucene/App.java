package a1.lucene;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
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

    private static void performSearch(Directory index, String queryString,
            CustomAnalyzer customAnalyzer) throws IOException, ParseException {
        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);

        // Use your custom analyzer here
        QueryParser queryParser = new QueryParser("processedText", customAnalyzer);

        Query query = queryParser.parse(queryString);

        TopDocs topDocs = searcher.search(query, 1000);
        ScoreDoc[] scoreDocs = topDocs.scoreDocs;

        saveResultsToFile(scoreDocs, searcher);
    }

    private static void saveResultsToFile(ScoreDoc[] scoreDocs, IndexSearcher searcher) {
        int queryNum = 0;
        int counter = 1;
        String runName = "run name";

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("Results.txt"))) {
            for (ScoreDoc scoreDoc : scoreDocs) {
                int docId = scoreDoc.doc;
                float score = scoreDoc.score;

                Document doc = searcher.doc(docId);
                String docIdString = doc.get("docId");

                // Write to the file with the actual score
                writer.write(queryNum + " Q0 " + docIdString + " " + counter + " " + score + " " + runName + "\n");
                counter++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException, ParseException {
        Directory index = FSDirectory.open(Paths.get("./index_dir"));

        CustomAnalyzer customAnalyzer = new CustomAnalyzer();

        performSearch(index, "Accusations of Cheating by Contractors on U.S. Defense Projects", customAnalyzer);
    }

}
