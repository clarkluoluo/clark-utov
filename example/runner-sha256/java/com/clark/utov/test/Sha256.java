package com.clark.utov.test;

public final class Sha256 {
    static {
        System.loadLibrary("sha256");
    }

    private Sha256() {}

    public static native byte[] hash(byte[] input);
}
