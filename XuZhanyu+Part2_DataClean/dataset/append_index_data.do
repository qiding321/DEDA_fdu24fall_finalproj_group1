*** Append Dataset:
global dir "/Users/kd/Downloads/FDU/博一上/讲座/DEDA Digital Economy & Decision Analytics/Project/freight_index"
cd "${dir}"
local files: dir . files "*.*"


foreach v in `files' {
dis "`v'"
}

local dtafiles: dir . files"*.dta"

local N = 1
foreach v in `dtafiles'{
use "`v'", clear
tempfile file`N'
save "`file`N''"
local N = `N' + 1
}
 
use "`file1'", clear
forvalues i = 2/404 {
append using "`file`i''"
}

keep if JAPAN!=.
sort date
drop n
bys date: gen n=_n
keep if n == 1
drop n
gen week = wofd(date)
format week %tw

save index, replace
use index, clear
cd "/Users/kd/Downloads/FDU/博一上/讲座/DEDA Digital Economy & Decision Analytics/Project"
export delimited using "/Users/kd/Downloads/FDU/博一上/讲座/DEDA Digital Economy & Decision Analytics/Project/index.csv", replace
cd "/Users/kd/Downloads/FDU/博一上/讲座/DEDA Digital Economy & Decision Analytics/Project"
use future, clear
merge m:1 week using index
drop _merge
save data, replace
export delimited using "/Users/kd/Downloads/FDU/博一上/讲座/DEDA Digital Economy & Decision Analytics/Project/data.csv", replace

