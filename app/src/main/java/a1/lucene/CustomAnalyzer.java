package main.java.a1.lucene;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.CharArraySet;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;

public class CustomAnalyzer extends Analyzer {

    // Set to store custom stop words
    private final Set<String> customStopWords;

    // Constructor loads custom stop words from file
    public CustomAnalyzer() throws IOException {
        this.customStopWords = loadStopWords();
    }

    // Override createComponents to define custom preprocessing
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        // Use StandardTokenizer as the source token stream
        StandardTokenizer source = new StandardTokenizer();

        // Apply custom stop words using StopFilter
        return new TokenStreamComponents(source, new StopFilter(source, CharArraySet.copy(customStopWords)));
    }

    // Load custom stop words from file
    private static Set<String> loadStopWords() throws IOException {
        Set<String> stopWords = new HashSet<>();

        // Get custome stop words form Stopwords.txt file
        try (BufferedReader reader = Files.newBufferedReader(Paths.get("./Stopwords.txt"))) {
            String line;
            // Read each line from the file and add it to the set of custome stopwords
            while ((line = reader.readLine()) != null) {
                stopWords.add(line.trim());
            }
        }
        return stopWords;
    }
}
