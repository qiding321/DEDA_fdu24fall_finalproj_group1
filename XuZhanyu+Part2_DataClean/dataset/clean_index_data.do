
	global dir "/Users/kd/Downloads/FDU/博一上/讲座/DEDA Digital Economy & Decision Analytics/Project/freight_index"
	cd "${dir}"
	filelist, dir("${dir}") pattern("*.dta") save(file_list) 
	use file_list, clear

	levelsof filename, local(files)
	foreach file of local files {
    use `file', clear

// Drop Observations we don't need:
	drop in 2
	drop in 3
	drop in 4
	drop in 5
	drop in 6
	drop in 7
	drop in 8
	drop in 9
	drop in 10
	drop in 11
	drop in 12
	drop in 13
// Rename Variables:
    ds
    rename `: word 1 of `r(varlist)'' line

// Drop Variables we don't need:
    ds
    local second_var `: word 2 of `r(varlist)''
    drop `second_var'

    ds
    local fourth_var `: word 3 of `r(varlist)''
    drop `fourth_var'

 // Rename the key Variable:
    ds
    local second_var `: word 2 of `r(varlist)''
    local new_name = "date" + substr("`second_var'", -8, 8)
    rename `second_var' `new_name'

    replace line = "CHINA" in 1

    gen line_cleaned = subinstr(line, "(", "", .)
    replace line_cleaned = subinstr(line_cleaned, ")", "", .)
    replace line_cleaned = subinstr(line_cleaned, " SERVICE", "", .)
    drop line
    rename line_cleaned line
	replace line = "WC_AMERICA" if line == "W/C AMERICA"
	replace line = "EC_AMERICA" if line == "E/C AMERICA"
	replace line = "SOUTHEAST_ASIA" if line == "SOUTHEAST ASIA"
	replace line = "AUSTRALIA_NEW_ZEALAND" if line == "AUSTRALIA/NEW ZEALAND"
	replace line = "SOUTH_AFRICA" if line == "SOUTH AFRICA"
	replace line = "SOUTH_AMERICA" if line == "SOUTH AMERICA"
	replace line = "WEST_EAST_AFRICA" if line == "WEST EAST AFRICA"
	replace line = "PERSIAN_GULF_RED_SEA" if line == "PERSIAN GULF/RED SEA"

    order line
	xpose, clear
	drop in 1
	
	rename v1 CHINA
	rename v2 JAPAN
	rename v3  EUROPE
	rename v4 WC_AMERICA
	rename v5 EC_AMERICA
	rename v6  KOREA
	rename v7 SOUTHEAST_ASIA
	rename v8  MEDITERRANEAN
	rename v9 AUSTRALIA_NEW_ZEALAND
	rename v10  SOUTH_AFRICA
	rename v11 SOUTH_AMERICA
	rename v12  WEST_EAST_AFRICA
	rename v13 PERSIAN_GULF_RED_SEA
	gen date = "`new_name'"
	replace date = subinstr(date, "date", "", .)
	gen date_temp = date(date, "YMD")
	format date_temp %td
	drop date
	rename date_temp date
	order date
    // Save Dataset:
    save `file', replace
}

// Delete the Temporary File List:
	erase file_list.dta

	
	
	
	
	
	
