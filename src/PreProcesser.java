
import java.util.HashSet;
import java.util.Set;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author ASUS-PC
 */
public class PreProcesser {

    public static final String FILTER = "ben sen o biz siz onlar ve veya böylece sa sn "
            + "böyle bazı bazen öyle en ile için bir de da hiç birçok herzaman her daha ne dek um "
            + "niçin sın niye mm im neden şöyle şey bu ki ise diğer başka sonra önce ya ön ancak";

    public static Set<String> filterTermSet(Set<String> termSet) {
        Set<String> newSet = new HashSet<String>();
        for (String t : termSet) {
            if (!FILTER.contains(t) && t.length() > 1) {
                newSet.add(t);
            }
            //System.out.println(t);
        }
        System.out.println("Removed term count :" + (termSet.size() - newSet.size()));
        return newSet;
    }
}
