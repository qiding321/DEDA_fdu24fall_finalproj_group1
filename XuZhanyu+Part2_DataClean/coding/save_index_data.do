// Define Directory Path:
	global dir "/Users/kd/Downloads/FDU/博一上/讲座/DEDA Digital Economy & Decision Analytics/Project/"
// Define a Path:
	local dir "${dir}data/sse_scrapy"

// Create dta file:
	capture mkdir "`dir'dta"
	shell ls "`dir'" > file_list.txt
	import delimited file_list.txt, clear

	describe
	rename v1 folder_name

// Iterate through each Date Folder:
	levelsof folder_name, local(folders)
	foreach folder of local folders {
		local date_dir "`dir'dta/`folder'"
		mkdir "`date_dir'"

		shell ls "`dir'/`folder'" > hour_files.txt

		import delimited hour_files.txt, clear
    
		describe
    
		rename v1 hour_file

		levelsof hour_file, local(hour_files)
		foreach hour of local hour_files {
			local csv_file "`dir'/`folder'/`hour'/中国出口集装箱运价指数.0.csv"
			capture confirm file "`csv_file'"
				if !_rc {
					import delimited "`csv_file'", clear
					save "`date_dir'/`hour'.dta", replace
        }
    }
}

// Delete the Temporary File List:
erase file_list.txt
erase hour_files.txt






