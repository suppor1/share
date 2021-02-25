import java.util.HashMap;
import java.util.Map;

public class Apptest {
    public static void main(String[] args) {
        System.out.println(fineMaxLen("ABCABCA"));
    }

    public static int fineMaxLen(String content) {
        int start =0;
        int maxLength = 0;

        Map<Character, Integer> reIndex = new HashMap<Character, Integer>();

        char[] chars = content.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            Integer lastIdx = reIndex.get(chars[i]);
            if (null != lastIdx && lastIdx >= start) {
                start = lastIdx +1;
            }

            if ((i -start + 1) > maxLength) {
                maxLength = i -start + 1;
            }

            reIndex.put(chars[i], i);
        }
        return maxLength;
    }
}