package assert_type

import "fmt"

type Duck interface {
	Quack()
}

type Cat struct {
	name string
}

func (c *Cat) Quack() {
}

type Mon struct {
	name string
}

func (m Mon) Quack() {

}

func NoMethodPointer() {
	var c interface{} = &Cat{name: "A"}
	switch c.(type) {
	case *Cat:
		cat := c.(*Cat)
		cat.Quack()
	default:
		fmt.Println("NoMethodPointer")
	}
}

func MethodPointer() {
	var c Duck = &Cat{name: "A"}
	switch c.(type) {
	case *Cat:
		cat := c.(*Cat)
		cat.Quack()
	default:
		fmt.Print("MethodPointer")
	}
}

func NoMethodStruct() {
	var m interface{} = Mon{name: "B"}
	switch m.(type) {
	case Mon:
		mon := m.(Mon)
		mon.Quack()
	default:
		fmt.Println("NoMethod")
	}
}

func MethodStruct() {
	var m Duck = Mon{name: "B"}
	switch m.(type) {
	case Mon:
		mon := m.(Mon)
		mon.Quack()
	default:
		fmt.Println("NoMethod")
	}
}
