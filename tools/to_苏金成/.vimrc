set nocompatible
source $VIMRUNTIME/vimrc_example.vim
"source $VIMRUNTIME/mswin.vim
"behave mswin

" set diffexpr=MyDiff()
" function MyDiff()
"   let opt = '-a --binary '
"   if &diffopt =~ 'icase' | let opt = opt . '-i ' | endif
"   if &diffopt =~ 'iwhite' | let opt = opt . '-b ' | endif
"   let arg1 = v:fname_in
"   if arg1 =~ ' ' | let arg1 = '"' . arg1 . '"' | endif
"   let arg2 = v:fname_new
"   if arg2 =~ ' ' | let arg2 = '"' . arg2 . '"' | endif
"   let arg3 = v:fname_out
"   if arg3 =~ ' ' | let arg3 = '"' . arg3 . '"' | endif
"   let eq = ''
"   if $VIMRUNTIME =~ ' '
"     if &sh =~ '\<cmd'
"       let cmd = '""' . $VIMRUNTIME . '\diff"'
"       let eq = '"'
"     else
"       let cmd = substitute($VIMRUNTIME, ' ', '" ', '') . '\diff"'
"     endif
"   else
"     let cmd = $VIMRUNTIME . '\diff'
"   endif
"   silent execute '!' . cmd . ' ' . opt . arg1 . ' ' . arg2 . ' > ' . arg3 . eq
" endfunction
colo  molokai
set go=m
set ru
set nu
set sm
set autoindent
set hlsearch
set ignorecase
set tabstop=4
set softtabstop=4
set shiftwidth=4
set expandtab
"set bg=dark
set autochdir
syntax on
filetype indent plugin on
set completeopt=menu,longest
set clipboard=unnamed
autocmd BufWinLeave *.* mkview
autocmd BufWinEnter *.* silent loadview 
"php ?? <?php ????????
let g:PHP_default_indenting = 1
"To indent 'case:' and 'default:' statements in switch() blocks: >
let g:PHP_vintage_case_default_indent = 1


let g:neocomplcache_enable_at_startup = 1 
inoremap <expr> <Tab> pumvisible() ? "<C-n>" : "\<C-g>u<Tab>" 
" inoremap  <expr> j pumvisible() ? "<down>" : "\<C-g>uj" 
" inoremap  <expr> k pumvisible() ? "<up>" : "\<C-g>uk" 
map <F3> :NERDTreeToggle<CR>
vmap <Enter> <Plug>(EasyAlign)
nmap <Leader>a <Plug>(EasyAlign)

nmap <F6> :colo random<CR>
"use pathogen to manage vim plugins
execute pathogen#infect()
"++++++++++++++++++++++++++++++2013/7/15+++++++++++++++++
"
"+++++++++++++++++++++++++++++2013/8/14/++++++++++++++++++++++
"????????????????
set fileencodings=utf-8
"+++++++++++++++++++++++++++++2013/8/16+++++++++++++++++++++++
"?????ɱ????ļ?
set nobackup

"+++++++++++++++++++++++++++++2013/9/05+++++++++++++++++++++
"????Ĭ??????
set guifont=Courier\ New\ 11
" set gfn=Consolas\ 11
" set guifont=consolas:h12:cANSI


" 2014/01/23 ???? 
" If buffer modified, update any 'Last modified: ' in the first 20 lines.
" 'Last modified: ' can have up to 10 characters before (they are retained).
" Restores cursor and window position using save_cursor variable.
" code hacked from: http://vim.wikia.com/wiki/Insert_current_date_or_time
function! LastModified()
  if &modified
    let save_cursor = getpos(".")
    let n = min([20, line("$")])
    keepjumps exe '1,' . n . 's#^\(.\{,10}Last modified:\).*#\1' .
		   \ strftime('%c %a') . '#e'
    call histdel('search', -1)
    call setpos('.', save_cursor)
  endif
endfun
autocmd BufWritePre * call LastModified()

