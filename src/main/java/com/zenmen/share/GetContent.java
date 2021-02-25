package com.zenmen.share;

public class GetContent {
    public static void main(String[] args) {
        Download dld = new Get1();
        Get1 dld1 = new Get1();
        System.out.println(dld.get("www.zenmen.com"));
        dld = new Get2();
        System.out.println(dld.get("www.zenmen.com"));
    }
}

class Get1 implements Download {
    public String get(String url) {
        return desc + " " + url;
    }
}

class Get2 implements Download {
    public String get(String url) {
        return desc + "[ "+ url + " ]";
    }
}

interface Download {
    String desc = "download url content";
    String get(String url);
}
