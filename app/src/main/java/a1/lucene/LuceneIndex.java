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

    private static Map<String, String> processDocument(String filepath) {
        try {
            File xmlFile = new File(filepath);

            org.jsoup.nodes.Document doc = Jsoup.parse(xmlFile, "UTF-8", "", org.jsoup.parser.Parser.xmlParser());

            Map<String, String> docData = new HashMap<>();

            List<Element> docList = new ArrayList<>(doc.getElementsByTag("doc"));

            for (Element docElement : docList) {
                String docId = docElement.getElementsByTag("docno").text().trim();
                String text = docElement.getElementsByTag("text").text().trim();
                docData.put(docId, text);
            }

            return docData;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static Map<String, String> processDocumentsInFolder(String folderPath) {
        Map<String, String> allDocData = new HashMap<>();

        File folder = new File(folderPath);

        FileFilter xmlFilter = file -> file.isFile();

        File[] xmlFiles = folder.listFiles(xmlFilter);

        if (xmlFiles == null) {
            System.out.println("here");
        }

        if (xmlFiles != null) {
            for (File xmlFile : xmlFiles) {
                Map<String, String> docData = processDocument(xmlFile.getAbsolutePath());
                if (docData != null) {
                    allDocData.putAll(docData);
                }
            }
        }

        return allDocData;
    }

    public static void buildIndex(Map<String, String> docData, String indexPath) {
        try {
            Directory index = FSDirectory.open(Paths.get(indexPath));
            CustomAnalyzer analyzer = new CustomAnalyzer();
            IndexWriterConfig config = new IndexWriterConfig(analyzer);
            IndexWriter writer = new IndexWriter(index, config);

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
        String folderPath = "./coll";
        Map<String, String> docData = processDocumentsInFolder(folderPath);
        buildIndex(docData, "./index_dir");
        IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get("./index_dir")));
        int numDocs = reader.numDocs();
    }
}
