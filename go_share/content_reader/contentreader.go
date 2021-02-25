package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
)

type ReaderUtil struct {

}

type ReaderContent interface {
	read(reader io.Reader) ([]byte, error)
}

func (r *ReaderUtil) read(reader io.Reader) ([]byte, error) {
	bytes, err := ioutil.ReadAll(reader)
	if nil != err {
		return nil, err
	}
	defer func() {
		switch reader.(type) {
		case io.ReadCloser:
			rc := reader.(io.ReadCloser)
			rc.Close()
		}
	}()
	return bytes, nil
}

//读取内容并且输出
func readerContent(file *os.File)  {
	content, err := ioutil.ReadAll(file)
	if nil != err {
		panic(err)
	}
	fmt.Println(string(content))
}

func main() {
	file, err := os.Open("go_share/content_reader/aac.txt")
	if nil != err {
		fmt.Println(err.Error())
		return
	}

	readerContent(file)
	fmt.Println("==========================")

	util := ReaderUtil{}
	//从文件读
	file, err = os.Open("go_share/content_reader/aac.txt")
	bytes, err := util.read(file)
	if nil != err {
		return
	}
	fmt.Println(string(bytes))

	//从字符串读
	bytes, err = util.read(strings.NewReader("read from string"))
	fmt.Println(string(bytes))

	//从网络读
	url := "http://news.baidu.com/"
	reader,err := fromUrl(&url)
	bytes, err = util.read(reader)
	if err != nil{
		fmt.Println(err)
	}
	fmt.Println(string(bytes))
}

func fromUrl(url *string) (io.Reader,error) {
	rsp, err := http.Get(*url)
	if nil != err {
		return nil, err
	}
	return rsp.Body,nil
}
