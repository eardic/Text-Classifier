
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import net.zemberek.erisim.Zemberek;
import net.zemberek.tr.yapi.TurkiyeTurkcesi;
import net.zemberek.yapi.Kelime;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author ASUS-PC
 */
public class DocReader {

    private final Zemberek zemberek = new Zemberek(new TurkiyeTurkcesi());
    private final File[] trainFiles, testFiles, files;
    private final FastVector attrList;
    private final Map<String, Double> idfMap = new HashMap<String, Double>();

    public DocReader(File[] trainFiles, File[] testFiles) {
        this.trainFiles = trainFiles;
        this.testFiles = testFiles;
        this.files = ArrayUtil.concat(trainFiles, testFiles);
        attrList = createTerms(this.files);
        System.err.println(attrList.size() + " Terms are created !");
        for (int i = 0; i < attrList.size(); ++i) {
            Attribute attr = (Attribute) attrList.elementAt(i);
            idfMap.put(attr.name(), 1.0);
        }
        System.err.println("IDF Values are calculated !");
    }

    private FastVector createTerms(File[] files) {
        try {
            Set<String> termSet = new HashSet<String>();
            for (File file : files) {
                BufferedReader reader = new BufferedReader(new FileReader(file));
                while (reader.ready()) {
                    String line = reader.readLine();
                    String[] words = line.split(" ");
                    for (String word : words) {
                        Kelime[] kelimeler = zemberek.kelimeCozumle(word);
                        if (kelimeler.length > 0) {
                            String kok = kelimeler[0].kok().icerik();
                            termSet.add(kok);
                        }
                    }
                }
                reader.close();
            }
            //System.err.println("\nAttribute:");
            FastVector terms = new FastVector();
            for (String term : termSet) {
                terms.addElement(new Attribute(term));
                // System.err.println(term + "-");
            }

            FastVector classValues = new FastVector();
            Set<String> classSet = new HashSet<String>();
            for (File file : files) {
                classSet.add(file.getName().substring(0, 3).toLowerCase());
            }
            //System.err.println("\nClass:");
            for (String category : classSet) {
                classValues.addElement(category);
                // System.out.print(category + "-");
            }
            terms.addElement(new Attribute("C", classValues));
            return terms;
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DocReader.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(DocReader.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }

    private double inverseDocFreq(String term) {
        try {
            double idf = 0;
            for (File file : files) {
                BufferedReader reader = new BufferedReader(new FileReader(file));
                while (reader.ready()) {
                    String line = reader.readLine();
                    String[] words = line.split(" ");
                    int freq = termFreq(term, words);
                    if (freq > 0) {
                        ++idf;
                        break;
                    }
                }
                reader.close();
            }
            //System.err.println("IDF: " + term + "=" + idf);
            return Math.log(files.length / idf) / Math.log(2);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DocReader.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(DocReader.class.getName()).log(Level.SEVERE, null, ex);
        }
        return Double.NaN;
    }

    public Instances createInstances() {
        Instances instances = new Instances("Train", attrList, trainFiles.length);
        for (File file : trainFiles) {
            Instance inst = createInstance(file);
            inst.setDataset(instances);
            instances.add(inst);
            instances.setClass((Attribute) attrList.lastElement());
        }
        return instances;
    }

    private int termFreq(String term, String[] words) {
        int f = 0;
        for (String word : words) {
            Kelime[] kelimeler = zemberek.kelimeCozumle(word);
            if (kelimeler.length > 0) {
                String kok = kelimeler[0].kok().icerik();
                if (term.equalsIgnoreCase(kok)) {
                    ++f;
                }
            }
        }
        return f;
    }

    public Instance createInstance(File file) {
        BufferedReader reader = null;
        try {
            Instance inst = new Instance(this.attrList.size());
            reader = new BufferedReader(new FileReader(file));
            Map<Attribute, Double> tFreqMap = new HashMap<Attribute, Double>();
            while (reader.ready()) {
                String line = reader.readLine();
                String[] words = line.split(" ");
                for (int i = 0; i < attrList.size(); ++i) {
                    Attribute attr = (Attribute) attrList.elementAt(i);
                    int tFreq = termFreq(attr.name(), words);
                    Double prevFreq = tFreqMap.get(attr);
                    tFreqMap.put(attr, ((prevFreq != null) ? (prevFreq + tFreq) : tFreq));
                }
            }
            normalizeVector(tFreqMap);
            // System.err.println("\nInstance:");
            for (Attribute attr : tFreqMap.keySet()) {
                inst.setValue(attr, tFreqMap.get(attr) * idfMap.get(attr.name()));
                //System.err.print(attr.name()+":"+inst.value(attr));
            }
            inst.setValue((Attribute) attrList.lastElement(), file.getName().substring(0, 3).toLowerCase());
            return inst;
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DocReader.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(DocReader.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException ex) {
                Logger.getLogger(DocReader.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return null;
    }

    private void normalizeVector(Map<Attribute, Double> vector) {
        // Normalization
        // Find max
        double max = Double.MIN_VALUE;
        for (Attribute attr : vector.keySet()) {
            if (vector.get(attr) > max) {
                max = vector.get(attr);
            }
            //System.err.println(attr.name() + ": " + vector.get(attr));
        }
        // Normalize
        for (Attribute attr : vector.keySet()) {
            vector.put(attr, vector.get(attr) / max);
        }
    }

    public void classify() {
        try {
            Instances testInstances = new Instances("Test", attrList, testFiles.length);
            for (File tFile : testFiles) {
                Instance inst = createInstance(tFile);
                inst.setDataset(testInstances);
                testInstances.setClass((Attribute) attrList.lastElement());
                NaiveBayes ibk = new NaiveBayes();
                ibk.buildClassifier(createInstances());
                double val = ibk.classifyInstance(inst);
                System.err.println(tFile.getName() + " => " + inst.classAttribute().value((int) val));
            }
        } catch (Exception ex) {
            Logger.getLogger(DocReader.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
