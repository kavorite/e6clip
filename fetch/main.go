package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"

	"github.com/anthonynsimon/bild/transform"
	"github.com/kolesa-team/go-webp/decoder"
	"github.com/kolesa-team/go-webp/encoder"
	"github.com/kolesa-team/go-webp/webp"
	"github.com/schollz/progressbar/v3"

	"github.com/jszwec/csvutil"
)

var allowFiles = map[string]struct{}{
	"jpg":  {},
	"jpeg": {},
	"png":  {},
	"webp": {},
}

var ipath, opath string

const (
	sourceBaseUrl = "https://static1.e621.net/data/"
	mirrorBaseUrl = "https://e6_dump.treehaus.dev/e6_dump/"
	baseUrl       = sourceBaseUrl
	userAgent     = "e6clip by kavorite"
	resize        = false
	width         = 224
	height        = 224
)

type setUserAgent struct {
	inner http.Transport
}

func (xport *setUserAgent) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Add("User-Agent", userAgent)
	return xport.inner.RoundTrip(req)
}

type record struct {
	ID  int    `csv:"id"`
	MD5 string `csv:"md5"`
	Ext string `csv:"file_ext"`
}

func fck(err error) {
	if err != nil {
		panic(err)
	}
}

func records(process func(record)) {
	istrm, err := os.Open(ipath)
	fck(err)
	total, err := istrm.Seek(0, io.SeekEnd)
	fck(err)
	_, err = istrm.Seek(0, io.SeekStart)
	fck(err)
	pbar := progressbar.DefaultBytes(total, fmt.Sprintf("read %s...", ipath))
	cursor := io.TeeReader(istrm, pbar)
	reader := csv.NewReader(cursor)
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
	var stem string
	var paths []string
	if baseUrl == sourceBaseUrl {
		stem = fmt.Sprintf("%s/%s/%s.%s", post.MD5[0:2], post.MD5[2:4], post.MD5, post.Ext)
		paths = []string{"sample/", ""}
	} else {
		stem = post.MD5 + ".webp"
		paths = []string{""}
	}
	out := fmt.Sprintf("%s/%d.webp", opath, post.ID)
	var (
		rsp *http.Response
		err error
	)
	for _, path := range paths {
		url := baseUrl + path + stem
		rsp, err = client.Get(url)
		if rsp.StatusCode == 200 {
			break
		}
	}
	fck(err)
	defer rsp.Body.Close()
	if rsp.StatusCode == 200 {
		dst, err := os.CreateTemp("", "")
		fck(err)
		if resize {
			img, err := webp.Decode(rsp.Body, &decoder.Options{})
			if err != nil {
				return
			}
			img = transform.Resize(img, width, height, transform.Linear)
			defer dst.Close()
			opt, err := encoder.NewLosslessEncoderOptions(encoder.PresetDrawing, 100)
			fck(err)
			err = webp.Encode(dst, img, opt)
			fck(err)
		} else {
			_, err = io.Copy(dst, rsp.Body)
			fck(err)
		}
		defer os.Rename(dst.Name(), out)
		fck(err)
	}
}

func main() {
	flag.StringVar(&ipath, "i", "", "Source csv file")
	flag.StringVar(&opath, "o", "posts", "Output directory")
	flag.Parse()
	if ipath == "" {
		fmt.Fprintf(os.Stderr, "missing required argument -i\n")
		return
	}

	conns := runtime.NumCPU() * 8
	xport := http.DefaultTransport.(*http.Transport).Clone()
	xport.MaxIdleConnsPerHost = conns
	xport.MaxIdleConns = conns
	xport.MaxConnsPerHost = conns
	client := &http.Client{Transport: &setUserAgent{*xport}}
	group := sync.WaitGroup{}
	group.Add(conns)
	defer group.Wait()
	queue := make(chan record)
	if err := os.Mkdir(opath, 0755); err != nil && !os.IsExist(err) {
		fck(err)
	}
	stat, err := os.Stat(opath)
	fck(err)
	if !stat.IsDir() {
		fmt.Fprintf(os.Stderr, "fatal: %s is not a directory\n", opath)
		return
	}
	for i := 0; i < conns; i++ {
		go func() {
			defer group.Done()
			for post := range queue {
				download(client, post)
			}
		}()
	}
	files, err := os.ReadDir(opath)
	fck(err)
	posts := make(map[int]struct{}, len(files))
	for _, entry := range files {
		if entry.Type().IsRegular() {
			f := entry.Name()
			id, err := strconv.Atoi(f[:len(f)-len(filepath.Ext(f))])
			fck(err)
			posts[id] = struct{}{}
		}
	}
	records(func(post record) {
		_, allow := allowFiles[post.Ext]
		_, exist := posts[post.ID]
		if allow && !exist {
			queue <- post
		}
	})
	close(queue)
}
