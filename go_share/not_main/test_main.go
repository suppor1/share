package notmain

type Test2 struct {

}

type Inf2 interface {
	Parse(url string) string
}

func (t *Test2) Parse(url string) string {
	return url
}

func Parse(url string) string {
	return url
}


