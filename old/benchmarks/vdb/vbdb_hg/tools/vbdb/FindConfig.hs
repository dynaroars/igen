
module FindConfig ( findConfigs ) where

import Note

import qualified Data.ByteString.Char8 as BC
import           Data.Char ( isSpace )
import           Data.Function ( on )
import           Data.List (  groupBy, sortBy )
import           Data.List.Split ( splitOn )
import           System.FilePath
import           System.Process ( callProcess, readProcess )

linuxgitdir :: FilePath
linuxgitdir = "/home/iago/tmp/linux/hunt"

gitCheckout :: Hash -> IO ()
gitCheckout hash = callProcess "/usr/bin/git"
  [ "--git-dir"
  , linuxgitdir </> ".git"
  , "--work-tree"
  , linuxgitdir
  , "checkout"
  , pp hash
  ]

findPattern :: Config -> String
findPattern k = "^(menu)?config " ++ pp k ++ "$"

findArgs :: [Config] -> [String]
findArgs ks =
  [ linuxgitdir
  , "-name"
  , "Kconfig*"
  , "-exec"
  , "egrep"
  , "-H"
  ]
  ++
  concat [ ["-e", findPattern k] | k <- ks]
  ++
  [ "{}"
  , ";"
  ]

rawFindConfigs :: [Config] -> IO String
rawFindConfigs ks = readProcess "/usr/bin/find" (findArgs ks) ""


readConfig :: String -> (Config,FilePath)
readConfig str = (cfg,path)
  where [kc_path,k_str] = splitOn ":" str
        path = dropFileName $ makeRelative linuxgitdir kc_path
        cfg  = BC.pack $ tail $ dropWhile (not . isSpace) k_str

readConfigs :: String -> [(Config,FilePath)]
readConfigs = map takePath . groupPaths . map readConfig . lines
  where groupPaths = groupBy ((==) `on` fst) . sortBy (compare `on` fst)
        takePath xs =
          case map snd xs of
               []  -> error "FindConfig.readConfigs: empty list"
               [p] -> (c,p)
               p:ps | all (`hpEq` p) ps -> (c,headPath p)
               p:_ -> (c, "???" ++ headPath p)
          where c = fst $ head xs
        hpEq = (==) `on` headPath

findConfigs :: Hash -> [Config] -> IO [FilePath]
findConfigs hash ks = do
  gitCheckout hash
  find_out <- rawFindConfigs ks
  let pathMap = readConfigs find_out
  return $ [ maybe "???" id (lookup k pathMap) | k <- ks ]

