package main

import "testing"

//go test -gcflags=-N -benchmem -test.count=3 -test.cpu=1 -test.benchtime=1s -bench=.

func BenchmarkDirectCall_struct_pointer(b *testing.B) {
	str := &str1{content: "www.zenmen.com"}
	for i := 0; i < b.N; i++ {
		str.Print()
	}
}

func BenchmarkDynamicDispatch_struct_pointer(b *testing.B) {
	str := test3(&str1{content: "www.zenmen.com"})
	for i := 0; i < b.N; i++ {
		str.Print()
	}
}

func BenchmarkDirectCall_struct(b *testing.B) {
	str := str2{content: "www.zenmen.com"}
	for i := 0; i < b.N; i++ {
		str.Print()
	}
}

func BenchmarkDynamicDispatch_struct(b *testing.B) {
	str := test3(str2{content: "www.zenmen.com"})
	for i := 0; i < b.N; i++ {
		str.Print()
	}
}
