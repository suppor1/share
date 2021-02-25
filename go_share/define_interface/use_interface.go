package main

import "fmt"

type Test1 struct {
	name string
}

func main111() {
	//变量定义
	var inf1 interface{} = Test1{name: "Test1"}
	fmt.Println(inf1)
}
//方法入参
func MethodParam(v interface{}) {
	if v == nil {
		fmt.Println("OK")
	} else {
		fmt.Println("Error")
	}
}
//方法返回
func MethodReturn() interface{} {

	return nil
}
