package main

type LeeCodeUtil struct {
}

type MaxLengthSubString interface {
	find(str string) int
}

func (l *LeeCodeUtil) find(str string) int {
	maxLength := 0
	startIndex := 0
	lastOccIndex := make(map[rune]int)

	bytes := []rune(str)

	for i, r := range bytes {
		if last, ok := lastOccIndex[r]; ok && last >= startIndex {
			startIndex = last + 1
		}

		if i-startIndex+1 > maxLength {
			maxLength = i - startIndex + 1
		}

		lastOccIndex[r] = i
	}

	return maxLength
}

func main() {
	lc := LeeCodeUtil{}

	println(lc.find("ABCABC"))
}
