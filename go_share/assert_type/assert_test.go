package assert_type

import "testing"

func BenchmarkNoMethodPointer(b *testing.B) {
	for i:=0; i<b.N; i++ {
		NoMethodPointer()
	}
}

func BenchmarkMethodPointer(b *testing.B) {
	for i:=0; i<b.N; i++ {
		MethodPointer()
	}
}

func BenchmarkNoMethod(b *testing.B) {
	for i:=0; i<b.N; i++ {
		NoMethodStruct()
	}
}

func BenchmarkMethod(b *testing.B) {
	for i:=0; i<b.N; i++ {
		MethodStruct()
	}
}

