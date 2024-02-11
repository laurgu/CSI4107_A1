package main.java.a1.lucene;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.ArrayList;
import java.util.Set;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.store.Directory;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class LuceneIndex {

    // Method to process an single XML file
    private static Map<String, String> processDocument(String filepath) {
        try {
            File xmlFile = new File(filepath);

            // Parse XML document using Jsoup
            org.jsoup.nodes.Document doc = Jsoup.parse(xmlFile, "UTF-8", "", org.jsoup.parser.Parser.xmlParser());

            // Hashmap will contain the docId and text of all docs within a file
            Map<String, String> docData = new HashMap<>();

            // Extract data from each 'doc' element in the XML
            List<Element> docList = new ArrayList<>(doc.getElementsByTag("doc"));

            // Extract the docId and text for each document and add them to the hashmap
            for (Element docElement : docList) {
                String docId = docElement.getElementsByTag("docno").text().trim();
                String text = docElement.getElementsByTag("text").text().trim().toLowerCase();
                docData.put(docId, text);
            }

            return docData;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // Method to process all XML documents in a folder and accumulate data
    public static Map<String, String> processDocumentsInFolder(String folderPath) {
        // Create a hashmap for all documents (docId + text)
        Map<String, String> allDocData = new HashMap<>();

        // Open folder
        File folder = new File(folderPath);

        // Extract all files from the specified folder
        File[] files = folder.listFiles();

        // Process each XML document and add document to the hashmap
        if (files != null) {
            for (File file : files) {
                Map<String, String> docData = processDocument(file.getAbsolutePath());
                if (docData != null) {
                    allDocData.putAll(docData);
                }
            }
        }

        return allDocData;
    }

    // Method to build the Lucene index from the processed document data
    public static void buildIndex(Map<String, String> docData, String indexPath) {
        try {
            // Open the Lucene index
            Directory index = FSDirectory.open(Paths.get(indexPath));
            CustomAnalyzer analyzer = new CustomAnalyzer();
            IndexWriterConfig config = new IndexWriterConfig(analyzer);
            IndexWriter writer = new IndexWriter(index, config);

            // Iterate over the document data and add to the Lucene index
            for (Map.Entry<String, String> entry : docData.entrySet()) {
                String docId = entry.getKey();
                String processedText = entry.getValue();

                Document doc = new Document();
                doc.add(new StringField("docId", docId, Field.Store.YES));
                doc.add(new TextField("processedText", processedText, Field.Store.YES));

                writer.addDocument(doc);
            }

            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        // Specify the folder containing XML documents
        String folderPath = "./coll";
        // Process XML documents and accumulate data
        Map<String, String> docData = processDocumentsInFolder(folderPath);
        // Build the Lucene index from the processed data
        buildIndex(docData, "./index_dir");
        // Open the Lucene index reader for further operations if needed
        IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get("./index_dir")));

        reader.close();
    }
}
