package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/schollz/progressbar/v3"

	"github.com/jszwec/csvutil"
)

var allowFiles = map[string]struct{}{
	"jpg":  {},
	"jpeg": {},
	"png":  {},
	"webp": {},
}

const userAgent = "e6clip by kavorite"

type setUserAgent struct {
	inner http.RoundTripper
}

func (xport *setUserAgent) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Add("User-Agent", userAgent)
	return xport.inner.RoundTrip(req)
}

const (
	opath   = "posts"
	baseUrl = "https://static1.e621.net/data"
)

type record struct {
	ID  string `csv:"id"`
	MD5 string `csv:"md5"`
	Ext string `csv:"file_ext"`
}

func fck(err error) {
	if err != nil {
		panic(err)
	}
}

func records(process func(record)) {
	istrm, err := os.Open("posts.csv")
	fck(err)
	reader := csv.NewReader(istrm)
	fck(err)
	dec, err := csvutil.NewDecoder(reader)
	fck(err)
	for {
		row := record{}
		if err := dec.Decode(&row); err == io.EOF {
			break
		}
		fck(err)
		process(row)
	}
}

func download(client *http.Client, post record) {
	stem := fmt.Sprintf("%s/%s/%s.%s", post.MD5[0:2], post.MD5[2:4], post.MD5, post.Ext)
	name := fmt.Sprintf("%s/%s.%s", opath, post.ID, post.Ext)
	if _, err := os.Stat(name); err == nil {
		return
	}
	for _, root := range []string{"sample", ""} {
		link := fmt.Sprintf("%s/%s/%s", baseUrl, root, stem)
		rsp, err := client.Get(link)
		fck(err)
		defer rsp.Body.Close()
		var dst io.Writer
		if rsp.StatusCode == 200 {
			ostrm, err := os.OpenFile(fmt.Sprintf("%s/%s.%s", opath, post.ID, post.Ext), os.O_CREATE|os.O_WRONLY, 0644)
			fck(err)
			dst = ostrm
			defer ostrm.Close()
		} else {
			dst = io.Discard
		}
		_, err = io.Copy(dst, rsp.Body)
		fck(err)
	}
}

func main() {
	conns := runtime.NumCPU() * 2
	xport := http.DefaultTransport.(*http.Transport).Clone()
	xport.MaxIdleConnsPerHost = conns
	xport.MaxIdleConns = conns
	xport.MaxConnsPerHost = conns
	client := &http.Client{Timeout: 10 * time.Second, Transport: &setUserAgent{xport}}
	group := sync.WaitGroup{}
	group.Add(conns)
	defer group.Wait()
	posts := make(chan record, conns)
	if err := os.Mkdir(opath, 0755); err != nil && !os.IsExist(err) {
		fck(err)
	}
	pbar := progressbar.Default(-1, "fetch posts...")
	for i := 0; i < conns; i++ {
		go func() {
			defer group.Done()
			for post := range posts {
				download(client, post)
			}
		}()
	}
	records(func(post record) {
		if _, ok := allowFiles[post.Ext]; ok {
			posts <- post
		}
		pbar.Add(1)
	})
	close(posts)
}
