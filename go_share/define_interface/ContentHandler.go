package main

import (
	"fmt"
)

var desc = "download url content"

type Download interface {
	Get(url string) string
}

type Download1 interface {
	Get(url string) string
}

type View interface {
	view(content string)
}

//结构体
type ContentHandler struct {
	content string
}

//方法体中没有使用接收者是，接收者也可以不指定名称
func (h ContentHandler) Get(url string) string {
	return desc + " [" + url + "]"
}

func (h ContentHandler) view(content string) {
	fmt.Println(content)
}

func main() {
	contentHandler := ContentHandler{}
	url := "www.zenmen.com"
	//download
	ctt := doDownload(url, contentHandler)
	contentHandler.content = ctt
 	//view
	doView(contentHandler)
	//接口方法
	//notmain.Parse("")

}
func doDownload(url string, ifc Download) string {
	return ifc.Get(url)
}
func doView(view View) {
	handler := view.(ContentHandler)
	view.view(handler.content)
}




