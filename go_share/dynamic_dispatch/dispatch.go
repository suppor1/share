package main

type test3 interface {
	Print()
}

type str1 struct {
	content string
}

func (s *str1) Print() {

}

type str2 struct {
	content string
}

func (s str2) Print() {

}