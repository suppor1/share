package composite_interface

type Retriever interface {
	Get(url string) string
}

type Poster interface {
	Post(url string,
		form map[string]string) string
}

func post(poster Poster) {
	poster.Post("www.zenmen.com", map[string]string{
		"name":"suppor",
		"learn":"golang",
	})
}

//golang 中也要这种 io.ReadWriteCloser
type RetrieverPoster interface {
	Retriever
	Poster
	//可以定义其他的方法
	//Socket(host string, port uint32)
}

func session(s RetrieverPoster)  {
	s.Get("")
	s.Post("", map[string]string{})
	//s.Socket("",1000)
}
