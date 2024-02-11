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

    private final Set<String> customStopWords;

    public CustomAnalyzer() throws IOException {
        this.customStopWords = loadStopWords();
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        StandardTokenizer source = new StandardTokenizer();
        return new TokenStreamComponents(source, new StopFilter(source, CharArraySet.copy(customStopWords)));
    }

    private static Set<String> loadStopWords() throws IOException {
        Set<String> stopWords = new HashSet<>();
        try (BufferedReader reader = Files.newBufferedReader(Paths.get("./Stopwords.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                stopWords.add(line.trim());
            }
        }
        return stopWords;
    }
}
