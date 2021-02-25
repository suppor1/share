package main

import (
	"encoding/json"
	"fmt"
)

type User1 struct {
	Name string
	Age  uint8
}

type User2 struct {
	Name string
	Age  uint8
}

//func (u User2) String() string {
//	builder := strings.Builder{}
//	builder.WriteString("{")
//	builder.WriteString(`"Name:"` + u.Name + ",")
//	builder.WriteString(`"Age:"` + strconv.Itoa(int(u.Age)))
//	builder.WriteString("}")
//	return builder.String()
//}

func main() {
	u1 := User1{Name: "Alice", Age: 1}
	u2 := User2{Name: "Bob", Age: 2}
	bytes, err := json.Marshal(u1)
	if nil != err {
		return
	}
	fmt.Println(string(bytes))
	fmt.Println(u2)

}
