package main

type User interface {
	PrintName()
}

type student struct {
	name string
}

func (s student) PrintName() {

}

type teacher struct {
	name string
}

func (t *teacher) PrintName() {

}

func main() {
	var st1 User = student{name: "Alice"}
	st1.PrintName()
	var st2 User = &student{name: "Bob"}
	st2.PrintName()

	var t1 User = &teacher{name: "Alice"}
	t1.PrintName()

	//var t2 User = teacher{name: "Bob"} //
	//t2.PrintName()

}
