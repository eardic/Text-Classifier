
import java.util.List;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.text.JTextComponent;
import net.zemberek.erisim.Zemberek;
import net.zemberek.tr.yapi.TurkiyeTurkcesi;
import net.zemberek.yapi.Kelime;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
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
public class DocClassifier {

    public static final String CLASS_ATTR_NAME = "Category";
    private final Zemberek zemberek = new Zemberek(new TurkiyeTurkcesi());
    private final File[] trainFiles, testFiles, files;
    private FastVector attrList, classValues;
    private final Map<String, Double> idfMap = new HashMap<String, Double>();
    private List<String> docPredList = new ArrayList<String>();

    public DocClassifier(File[] trainFiles, File[] testFiles) {
        this.trainFiles = trainFiles;
        this.testFiles = testFiles;
        this.files = ArrayUtil.concat(trainFiles, testFiles);
        this.attrList = createTerms(this.files);
        System.err.println(attrList.size() + " Terms are created !");
    }

    public FastVector getClassValues() {
        return classValues;
    }

    public List<String> getDocPredList() {
        return docPredList;
    }

    private FastVector createTerms(File[] files) {
        try {
            Set<String> termSet = new HashSet<String>();
            for (File file : files) {
                BufferedReader reader = new BufferedReader(new FileReader(file));
                Set<String> docTermSet = new HashSet<String>();
                while (reader.ready()) {
                    String line = reader.readLine();
                    String[] words = line.split(" ");
                    for (String word : words) {
                        Kelime[] kelimeler = this.zemberek.kelimeCozumle(word);
                        if (kelimeler.length > 0) {
                            String kok = kelimeler[0].kok().icerik();
                            docTermSet.add(kok);
                            termSet.add(kok);
                        }
                    }
                }
                // DF for a doc
                for (String t : docTermSet) {
                    Double freq = this.idfMap.get(t);
                    this.idfMap.put(t, ((freq != null) ? (freq + 1) : 1));
                }
                reader.close();
            }
            //Remove some words like ve,veya,de,da,in from set
            termSet = PreProcesser.filterTermSet(termSet);
            //IDF Calculation
            for (String t : termSet) {
                Double df = this.idfMap.get(t);
                if (df != null) {
                    this.idfMap.put(t, Math.log(files.length / df) / Math.log(2));
                } else {
                    this.idfMap.put(t, 0.0);
                }
                //System.out.println(t + ": " + df);
            }
            // Attribute creation
            //System.err.println("\nAttribute:");
            FastVector terms = new FastVector();
            for (String term : termSet) {
                terms.addElement(new Attribute(term));
                // System.err.println(term + "-");
            }
            // Class values are created
            Set<String> classSet = new HashSet<String>();
            for (File file : files) {
                classSet.add(file.getName().substring(0, 3).toLowerCase());
            }
            //System.err.println("\nClass:");
            this.classValues = new FastVector();
            for (String category : classSet) {
                this.classValues.addElement(category);
                // System.out.print(category + "-");
            }
            terms.addElement(new Attribute(CLASS_ATTR_NAME, classValues));
            return terms;
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DocClassifier.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(DocClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }

    public Instances createInstances(File[] files) {
        Instances instances = new Instances("Inst" + files.hashCode(), attrList, files.length);
        for (File file : files) {
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
                for (int i = 0; i < attrList.size() - 1; ++i) {
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
            Logger.getLogger(DocClassifier.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(DocClassifier.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException ex) {
                Logger.getLogger(DocClassifier.class.getName()).log(Level.SEVERE, null, ex);
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

    public String performanceEval(Evaluation ev) {
        String val = "AVERAGE PERFORMANCES\n";
        val += "TPR\t: " + ev.weightedTruePositiveRate() + "\n";
        val += "TNR\t: " + ev.weightedTrueNegativeRate() + "\n";
        val += "FPR\t: " + ev.weightedFalsePositiveRate() + "\n";
        val += "FNR\t: " + ev.weightedFalseNegativeRate() + "\n";
        val += "Precision\t: " + ev.weightedPrecision() + "\n";
        val += "Recall\t: " + ev.weightedRecall() + "\n";
        val += "F-Measure\t: " + ev.weightedFMeasure() + "\n";
        return val;
    }

    public Evaluation classify(Classifier classifier) throws Exception {
        docPredList.clear();
        Instances testInstances = createInstances(testFiles);
        Instances trainInstances = createInstances(trainFiles);
        classifier.buildClassifier(trainInstances);
        Evaluation ev = new Evaluation(trainInstances);
        for (int i = 0; i < testInstances.numInstances(); ++i) {
            Instance inst = testInstances.instance(i);
            double pred = ev.evaluateModelOnceAndRecordPrediction(classifier, inst);
            docPredList.add(testFiles[i].getName() + "\t=>\t" + inst.classAttribute().value((int) pred));
        }
        return ev;
    }

    public Evaluation cvClassify(Classifier classifier, int k) throws Exception {
        docPredList.clear();
        Instances trainInstances = createInstances(trainFiles);
        Evaluation ev = new Evaluation(trainInstances);
        ev.crossValidateModel(classifier, trainInstances, k, new Random(1));
        return ev;
    }
}
